"""Tests for the paired energy-saving experiment chain.

Run with::

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import tempfile
import unittest
from fractions import Fraction as F

from power_billing import (
    ExperimentRegistration,
    ExperimentStore,
    Segment,
    SiteRegistration,
    TrajectoryStore,
    build_report,
    compute_site_result,
    handover_result_at,
)
from power_billing.errors import (
    ConflictError,
    ExperimentError,
    NotFoundError,
)


def seg(sid, start, end, power):
    return Segment(sid, F(start), F(end), F(power))


def make_store(traces):
    """traces: {trace_id: (revision, [Segment, ...])}"""
    store = TrajectoryStore()
    for trace_id, (revision, segments) in traces.items():
        store.register(trace_id, revision, segments)
    return store


def main_traces():
    """The one-hour equal-length scenario: pairs (6,2;2,2) and (4,2;2,2)."""
    return {
        "A": ("a1", [seg("A0", 0, 1, 6), seg("A1", 1, 2, 2)]),
        "B": ("b1", [seg("B0", 0, 2, 2)]),
        "C": ("c1", [seg("C0", 0, 1, 4), seg("C1", 1, 2, 2)]),
        "D": ("d1", [seg("D0", 0, 2, 2)]),
    }


def hourly_site(site_id, trace_id):
    return SiteRegistration(
        site_id=site_id,
        site_version="v1",
        baselines=((F(0), F(1)),),
        observation=(F(1), F(2)),
        trace_id=trace_id,
    )


def register_main_sites(exp_store):
    for sid in ("A", "B", "C", "D"):
        exp_store.register_site(hourly_site(sid, sid))


def compute_all(exp_store, traj_store, site_ids=("A", "B", "C", "D")):
    return {
        sid: exp_store.compute_and_store_site_result(sid, traj_store)
        for sid in site_ids
    }


def exp1_registration():
    """Two pairs, one independent sign group per pair."""
    return ExperimentRegistration.from_json({
        "experiment_id": "EXP1",
        "sign_groups": ["g1", "g2"],
        "observed_faces": {"g1": "+", "g2": "+"},
        "pairs": [
            {"pair_id": "p1", "left": "A", "right": "B",
             "group": "g1", "wiring": "same", "observed_treated": "left"},
            {"pair_id": "p2", "left": "C", "right": "D",
             "group": "g2", "wiring": "same", "observed_treated": "left"},
        ],
    })


def exp2_registration():
    """The same two pairs wired same-direction onto one shared sign."""
    return ExperimentRegistration.from_json({
        "experiment_id": "EXP2",
        "sign_groups": ["g"],
        "observed_faces": {"g": "+"},
        "pairs": [
            {"pair_id": "p1", "left": "A", "right": "B",
             "group": "g", "wiring": "same", "observed_treated": "left"},
            {"pair_id": "p2", "left": "C", "right": "D",
             "group": "g", "wiring": "same", "observed_treated": "left"},
        ],
    })


class SiteResultTests(unittest.TestCase):
    def test_equal_length_one_hour_results(self):
        store = make_store(main_traces())
        results = {
            sid: compute_site_result(hourly_site(sid, sid), store)
            for sid in ("A", "B", "C", "D")
        }
        self.assertEqual(results["A"].result, F(4))   # 6*1 - 2
        self.assertEqual(results["B"].result, F(0))   # 2*1 - 2
        self.assertEqual(results["C"].result, F(2))   # 4*1 - 2
        self.assertEqual(results["D"].result, F(0))
        for res in results.values():
            self.assertEqual(res.status, "ok")
            self.assertEqual(res.baseline_duration, F(1))
            self.assertEqual(res.observation_duration, F(1))

    def test_unequal_baseline_periods_use_weighted_mean(self):
        # Baselines [0,1) at 8 and [1,3) at 2: the mean power is the
        # energy-weighted (8*1 + 2*2)/3 = 4, not the average of means 5.
        store = make_store({
            "E": ("e1", [seg("E0", 0, 1, 8), seg("E1", 1, 3, 2),
                         seg("E2", 3, 4, 2)]),
        })
        reg = SiteRegistration(
            "E", "v1",
            baselines=((F(0), F(1)), (F(1), F(3))),
            observation=(F(3), F(4)),
            trace_id="E",
        )
        res = compute_site_result(reg, store)
        self.assertEqual(res.status, "ok")
        self.assertEqual(res.baseline_energy, F(12))
        self.assertEqual(res.baseline_duration, F(3))
        self.assertEqual(res.baseline_mean_power, F(4))
        self.assertEqual(res.result, F(2))  # 4*1 - 2, not 5 - 2 = 3

    def test_unequal_end_durations_do_not_subtract_energies(self):
        # Baseline 1h at 6, observation 2h at 2: result is 6*2 - 4 = 8;
        # subtracting energies directly would wrongly give 6 - 4 = 2.
        store = make_store({
            "E": ("e1", [seg("E0", 0, 1, 6), seg("E1", 1, 3, 2)]),
        })
        reg = SiteRegistration(
            "E", "v1",
            baselines=((F(0), F(1)),),
            observation=(F(1), F(3)),
            trace_id="E",
        )
        res = compute_site_result(reg, store)
        self.assertEqual(res.result, F(8))
        self.assertEqual(res.expected_observation, F(12))
        self.assertEqual(res.observation_energy, F(4))

    def test_period_boundaries_inside_segments_are_exact(self):
        store = make_store({"S": ("s1", [seg("S0", 0, 2, 4)])})
        reg = SiteRegistration(
            "S", "v1",
            baselines=((F(1, 2), F(3, 2)),),
            observation=(F(3, 2), F(2)),
            trace_id="S",
        )
        res = compute_site_result(reg, store)
        self.assertEqual(res.baseline_energy, F(4))
        self.assertEqual(res.observation_energy, F(2))
        self.assertEqual(res.observation_duration, F(1, 2))
        self.assertEqual(res.result, F(0))  # 4 * 1/2 - 2

    def test_zero_power_is_a_measurement(self):
        # Power 0 over the observation is measured data, not a gap.
        store = make_store({
            "F0": ("f1", [seg("F00", 0, 1, 4), seg("F01", 1, 2, 0)]),
        })
        res = compute_site_result(hourly_site("F0", "F0"), store)
        self.assertEqual(res.status, "ok")
        self.assertEqual(res.observation_energy, F(0))
        self.assertEqual(res.result, F(4))  # 4*1 - 0

    def test_no_measurement_is_missing_evidence(self):
        # The observation period has no segment at all: a gap, not a zero.
        store = make_store({"F1": ("f1", [seg("F10", 0, 1, 4)])})
        res = compute_site_result(hourly_site("F1", "F1"), store)
        self.assertEqual(res.status, "missing_evidence")
        self.assertIsNone(res.result)
        self.assertEqual(len(res.gaps), 1)
        gap = res.gaps[0]
        self.assertEqual((gap["start"], gap["end"]), (F(1), F(2)))
        self.assertEqual(gap["meter"], "F1")
        self.assertEqual(gap["period"], (F(1), F(2)))

    def test_gap_inside_baseline_reported(self):
        store = make_store({
            "G": ("g1", [seg("G0", 0, F(1, 2), 4), seg("G1", F(3, 4), 2, 4)]),
        })
        res = compute_site_result(hourly_site("G", "G"), store)
        self.assertEqual(res.status, "missing_evidence")
        # Only the baseline period [0,1) has a gap; [1,2) is covered.
        self.assertEqual(
            [(g["start"], g["end"], g["period"]) for g in res.gaps],
            [(F(1, 2), F(3, 4), (F(0), F(1)))],
        )

    def test_overlapping_and_empty_periods_rejected(self):
        with self.assertRaises(ExperimentError):
            SiteRegistration(
                "X", "v1",
                baselines=((F(0), F(2)), (F(1), F(3))),
                observation=(F(3), F(4)),
                trace_id="X",
            ).validate()
        with self.assertRaises(ExperimentError):
            SiteRegistration(
                "X", "v1", baselines=((F(0), F(0)),), observation=(F(1), F(2)),
                trace_id="X",
            ).validate()
        with self.assertRaises(ExperimentError):
            SiteRegistration(
                "X", "v1", baselines=((F(0), F(1)),), observation=(F(0), F(2)),
                trace_id="X",
            ).validate()


class HandoverTests(unittest.TestCase):
    def handover_store(self, old_segs, new_segs):
        return make_store({
            "HOLD": ("h1", old_segs),
            "HNEW": ("h1", new_segs),
        })

    def spec_registration(self, low, up):
        return SiteRegistration(
            "H", "v1",
            baselines=((F(0), F(1)),),
            observation=(F(1), F(2)),
            old_trace_id="HOLD", new_trace_id="HNEW",
            handover=(F(low), F(up)),
        )

    def test_spec_example_is_ambiguous(self):
        # Baseline [0,1), observation [1,2), old meter constant 2, new
        # constant 1, cut point anywhere in [0,2]: t=0 or t=2 give
        # difference 0, t=1 gives 1 -> not adoptable.
        store = self.handover_store([seg("HO0", 0, 2, 2)], [seg("HN0", 0, 2, 1)])
        reg = self.spec_registration(0, 2)
        res = compute_site_result(reg, store)
        self.assertEqual(res.status, "ambiguous_handover")
        self.assertIsNone(res.result)
        self.assertEqual(
            res.cut_points, ((F(0), F(0)), (F(1), F(1)))
        )
        old, new = store.trajectory("HOLD"), store.trajectory("HNEW")
        self.assertEqual(handover_result_at(reg, old, new, F(0)), F(0))
        self.assertEqual(handover_result_at(reg, old, new, F(1)), F(1))
        self.assertEqual(handover_result_at(reg, old, new, F(2)), F(0))

    def test_unique_when_meters_agree_inside_handover(self):
        # New meter reads the same 2 as the old one across the whole
        # handover interval [1/2, 1]; only after 3/2 does it drop to 1.
        store = self.handover_store(
            [seg("HO0", 0, 2, 2)],
            [seg("HN0", 0, F(3, 2), 2), seg("HN1", F(3, 2), 2, 1)],
        )
        res = compute_site_result(self.spec_registration(F(1, 2), F(1)), store)
        self.assertEqual(res.status, "ok")
        self.assertEqual(res.result, F(1, 2))  # 2*1 - 3/2
        self.assertEqual(res.baseline_energy, F(2))
        self.assertEqual(res.observation_energy, F(3, 2))

    def test_handover_outside_required_periods(self):
        # The switch happens in [2,3], after every required period: only
        # the old meter's readings are ever used.
        store = self.handover_store([seg("HO0", 0, 2, 2)], [seg("HN0", 0, 3, 1)])
        res = compute_site_result(self.spec_registration(2, 3), store)
        self.assertEqual(res.status, "ok")
        self.assertEqual(res.result, F(0))  # all-old: 2*1 - 2

    def test_known_cut_point_is_degenerate_interval(self):
        store = self.handover_store([seg("HO0", 0, 2, 2)], [seg("HN0", 0, 2, 1)])
        res = compute_site_result(self.spec_registration(1, 1), store)
        self.assertEqual(res.status, "ok")
        self.assertEqual(res.result, F(1))  # baseline old 2, observation new 1

    def test_missing_old_reading_inside_handover(self):
        # Old meter has a gap [1/2, 3/4) inside [L,U] = [1/4, 1]; that
        # interval could actually be used, so it is missing evidence.
        store = self.handover_store(
            [seg("HO0", 0, F(1, 2), 2), seg("HO1", F(3, 4), 2, 2)],
            [seg("HN0", 0, 2, 1)],
        )
        res = compute_site_result(
            self.spec_registration(F(1, 4), F(1)), store
        )
        self.assertEqual(res.status, "missing_evidence")
        self.assertEqual(len(res.gaps), 1)
        self.assertEqual(res.gaps[0]["meter"], "HOLD")
        self.assertEqual((res.gaps[0]["start"], res.gaps[0]["end"]),
                         (F(1, 2), F(3, 4)))

    def test_missing_new_reading_inside_handover(self):
        # New meter stops at 3/2 but the observation runs to 2.
        store = self.handover_store(
            [seg("HO0", 0, 2, 2)],
            [seg("HN0", 0, F(3, 2), 1)],
        )
        res = compute_site_result(
            self.spec_registration(F(1, 4), F(1, 2)), store
        )
        self.assertEqual(res.status, "missing_evidence")
        self.assertEqual(res.gaps[0]["meter"], "HNEW")
        self.assertEqual((res.gaps[0]["start"], res.gaps[0]["end"]),
                         (F(3, 2), F(2)))

    def test_handover_validation(self):
        with self.assertRaises(ExperimentError):
            SiteRegistration(
                "H", "v1", baselines=((F(0), F(1)),), observation=(F(1), F(2)),
                old_trace_id="HOLD", new_trace_id="HNEW",
                handover=(F(2), F(1)),  # L > U
            ).validate()
        with self.assertRaises(ExperimentError):
            SiteRegistration(
                "H", "v1", baselines=((F(0), F(1)),), observation=(F(1), F(2)),
                trace_id="T", old_trace_id="HOLD", new_trace_id="HNEW",
                handover=(F(0), F(1)),
            ).validate()


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.traj = make_store(main_traces())
        self.exp = ExperimentStore()
        register_main_sites(self.exp)
        self.results = compute_all(self.exp, self.traj)

    def test_independent_signs_statistic_and_probability(self):
        self.exp.register_experiment(exp1_registration())
        doc = self.exp.publish_report("R1", "EXP1")
        self.assertEqual(doc["status"], "ok")
        self.assertEqual(doc["observed_statistic"], {"n": 6, "d": 1})
        self.assertEqual(doc["p_value"], {"n": 1, "d": 4})
        self.assertEqual(doc["assignments_total"], 4)
        self.assertEqual(
            [(d["statistic"], d["multiplicity"]) for d in doc["null_distribution"]],
            [({"n": -6, "d": 1}, 1), ({"n": -2, "d": 1}, 1),
             ({"n": 2, "d": 1}, 1), ({"n": 6, "d": 1}, 1)],
        )
        contributions = {p["pair_id"]: p["contribution"] for p in doc["pairs"]}
        self.assertEqual(contributions, {"p1": {"n": 4, "d": 1},
                                         "p2": {"n": 2, "d": 1}})

    def test_shared_sign_same_direction(self):
        self.exp.register_experiment(exp2_registration())
        doc = self.exp.publish_report("R2", "EXP2")
        self.assertEqual(doc["observed_statistic"], {"n": 6, "d": 1})
        self.assertEqual(doc["p_value"], {"n": 1, "d": 2})
        self.assertEqual(doc["assignments_total"], 2)
        self.assertEqual(
            [(d["statistic"], d["multiplicity"]) for d in doc["null_distribution"]],
            [({"n": -6, "d": 1}, 1), ({"n": 6, "d": 1}, 1)],
        )

    def test_correction_produces_new_result_version_and_report(self):
        self.exp.register_experiment(exp1_registration())
        r1 = self.exp.publish_report("R1", "EXP1")
        self.assertEqual(r1["sites"]["A"]["result_version"], "A#r1")

        # Correct pair 1's left site observation power 2 -> 6.
        self.traj.revise_json("A", {
            "revision": "a2",
            "replace_segment": {"segment_id": "A1", "start": 1, "end": 2,
                                "power": 6},
        })
        res_a2 = self.exp.compute_and_store_site_result("A", self.traj)
        self.assertEqual(res_a2.result_version, "A#r2")
        self.assertEqual(res_a2.result, F(0))  # 6*1 - 6
        self.assertEqual(res_a2.meters["revision"], "a2")

        r3 = self.exp.publish_report("R3", "EXP1")
        self.assertEqual(r3["sites"]["A"]["result_version"], "A#r2")
        self.assertEqual(r3["observed_statistic"], {"n": 2, "d": 1})
        self.assertEqual(r3["p_value"], {"n": 1, "d": 2})
        # Repeated statistic values still count their sign faces.
        self.assertEqual(
            [(d["statistic"], d["multiplicity"]) for d in r3["null_distribution"]],
            [({"n": -2, "d": 1}, 2), ({"n": 2, "d": 1}, 2)],
        )

        # The already published R1 still points back at the version it used.
        r1_back = self.exp.get_report("R1")
        self.assertEqual(r1_back["sites"]["A"]["result_version"], "A#r1")
        self.assertEqual(r1_back["observed_statistic"], {"n": 6, "d": 1})
        self.assertTrue(self.exp.verify_report("R1"))
        self.assertTrue(self.exp.verify_report("R3"))

    def test_reverse_wiring_flips_the_pair(self):
        # One shared sign; pair 2 wired reverse, so face '+' treats its
        # right site.  Observed: p1 left, p2 right -> T = 4 - 2 = 2.
        reg = ExperimentRegistration.from_json({
            "experiment_id": "EXP3",
            "sign_groups": ["g"],
            "observed_faces": {"g": "+"},
            "pairs": [
                {"pair_id": "p1", "left": "A", "right": "B",
                 "group": "g", "wiring": "same", "observed_treated": "left"},
                {"pair_id": "p2", "left": "C", "right": "D",
                 "group": "g", "wiring": "reverse", "observed_treated": "right"},
            ],
        })
        self.exp.register_experiment(reg)
        doc = self.exp.publish_report("R1", "EXP3")
        self.assertEqual(doc["observed_statistic"], {"n": 2, "d": 1})
        self.assertEqual(doc["p_value"], {"n": 1, "d": 2})
        self.assertEqual(
            [(d["statistic"], d["multiplicity"]) for d in doc["null_distribution"]],
            [({"n": -2, "d": 1}, 1), ({"n": 2, "d": 1}, 1)],
        )

    def test_missing_evidence_invalidates_without_dropping_pairs(self):
        # F1's observation is unmeasured; the report must say so and must
        # not silently continue with the remaining pair.
        self.traj.register("F1", "f1", [seg("F10", 0, 1, 4)])
        self.exp.register_site(hourly_site("F1", "F1"))
        self.exp.compute_and_store_site_result("F1", self.traj)
        reg = ExperimentRegistration.from_json({
            "experiment_id": "EXP4",
            "sign_groups": ["g1", "g2"],
            "observed_faces": {"g1": "+", "g2": "+"},
            "pairs": [
                {"pair_id": "p1", "left": "A", "right": "B",
                 "group": "g1", "wiring": "same", "observed_treated": "left"},
                {"pair_id": "p2", "left": "F1", "right": "D",
                 "group": "g2", "wiring": "same", "observed_treated": "left"},
            ],
        })
        self.exp.register_experiment(reg)
        doc = self.exp.publish_report("R4", "EXP4")
        self.assertEqual(doc["status"], "invalid")
        self.assertIsNone(doc["observed_statistic"])
        self.assertIsNone(doc["p_value"])
        self.assertIsNone(doc["assignments_total"])
        self.assertEqual(doc["null_distribution"], [])
        reasons = [p["site_id"] for p in doc["problems"] if "site_id" in p]
        self.assertEqual(reasons, ["F1"])
        self.assertIn("missing_evidence", doc["problems"][0]["reason"])
        self.assertEqual(doc["problems"][0]["gaps"][0]["start"], {"n": 1, "d": 1})
        # The frozen pair count is untouched; nothing was dropped.
        self.assertEqual(len(doc["sign_groups"]), 2)
        self.assertTrue(self.exp.verify_report("R4"))

    def test_observed_assignment_violating_wiring_invalidates(self):
        # Face '+' with 'same' wiring treats the left site; registering
        # 'right' as the observed treatment contradicts the wiring.
        reg = ExperimentRegistration.from_json({
            "experiment_id": "EXP5",
            "sign_groups": ["g1", "g2"],
            "observed_faces": {"g1": "+", "g2": "+"},
            "pairs": [
                {"pair_id": "p1", "left": "A", "right": "B",
                 "group": "g1", "wiring": "same", "observed_treated": "left"},
                {"pair_id": "p2", "left": "C", "right": "D",
                 "group": "g2", "wiring": "same", "observed_treated": "right"},
            ],
        })
        self.exp.register_experiment(reg)
        doc = self.exp.publish_report("R5", "EXP5")
        self.assertEqual(doc["status"], "invalid")
        self.assertIsNone(doc["observed_statistic"])
        problem = doc["problems"][0]
        self.assertEqual(problem["pair_id"], "p2")
        self.assertIn("wiring", problem["reason"])
        self.assertEqual(problem["observed_treated"], "right")
        self.assertEqual(problem["derived_treated"], "left")

    def test_sign_group_limit_is_twelve(self):
        groups = [f"g{i}" for i in range(13)]
        with self.assertRaises(ExperimentError):
            ExperimentRegistration.from_json({
                "experiment_id": "TOO_MANY",
                "sign_groups": groups,
                "observed_faces": {g: "+" for g in groups},
                "pairs": [{"pair_id": "p1", "left": "A", "right": "B",
                           "group": "g0", "wiring": "same",
                           "observed_treated": "left"}],
            })
        # Twelve is fine.
        ok_groups = [f"g{i}" for i in range(12)]
        reg = ExperimentRegistration.from_json({
            "experiment_id": "TWELVE",
            "sign_groups": ok_groups,
            "observed_faces": {g: "+" for g in ok_groups},
            "pairs": [{"pair_id": "p1", "left": "A", "right": "B",
                       "group": "g0", "wiring": "same",
                       "observed_treated": "left"}],
        })
        self.assertEqual(len(reg.sign_groups), 12)

    def test_experiment_registration_is_frozen(self):
        self.exp.register_experiment(exp1_registration())
        again = self.exp.register_experiment(exp1_registration())
        self.assertEqual(again["experiment_id"], "EXP1")
        changed = ExperimentRegistration.from_json({
            "experiment_id": "EXP1",
            "sign_groups": ["g1", "g2"],
            "observed_faces": {"g1": "+", "g2": "-"},
            "pairs": [
                {"pair_id": "p1", "left": "A", "right": "B",
                 "group": "g1", "wiring": "same", "observed_treated": "left"},
                {"pair_id": "p2", "left": "C", "right": "D",
                 "group": "g2", "wiring": "same", "observed_treated": "right"},
            ],
        })
        with self.assertRaises(ConflictError):
            self.exp.register_experiment(changed)

    def test_site_registration_versions_and_conflicts(self):
        doc = self.exp.get_site_registration("A")
        self.assertEqual(doc["site_version"], "v1")
        with self.assertRaises(ConflictError):
            self.exp.register_site(SiteRegistration(
                "A", "v1", baselines=((F(0), F(2)),), observation=(F(2), F(3)),
                trace_id="A",
            ))
        # A new version is fine and becomes current.
        self.exp.register_site(SiteRegistration(
            "A", "v2", baselines=((F(0), F(1)),), observation=(F(1), F(2)),
            trace_id="A",
        ))
        self.assertEqual(self.exp.get_site_registration("A")["site_version"], "v2")
        self.assertEqual(
            self.exp.get_site_registration("A", "v1")["site_version"], "v1")

    def test_site_result_idempotent_without_correction(self):
        again = self.exp.compute_and_store_site_result("A", self.traj)
        self.assertEqual(again.result_version, "A#r1")
        self.assertEqual(len(self.exp.site_results("A")), 1)

    def test_report_replay_and_conflict(self):
        self.exp.register_experiment(exp1_registration())
        doc = self.exp.publish_report("R1", "EXP1")
        replay = self.exp.publish_report("R1", "EXP1")
        self.assertEqual(replay, doc)
        # Same id, different content (different experiment) -> conflict.
        self.exp.register_experiment(exp2_registration())
        with self.assertRaises(ConflictError):
            self.exp.publish_report("R1", "EXP2")

    def test_unknown_experiment_and_report(self):
        with self.assertRaises(NotFoundError):
            self.exp.publish_report("RX", "NOPE")
        with self.assertRaises(NotFoundError):
            self.exp.verify_report("NOPE")
        with self.assertRaises(NotFoundError):
            self.exp.compute_and_store_site_result("NOPE", self.traj)

    def test_reopening_a_database_preserves_the_chain(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "exp.db")
            store = ExperimentStore(path)
            register_main_sites(store)
            traj = make_store(main_traces())
            compute_all(store, traj)
            store.register_experiment(exp1_registration())
            store.publish_report("R1", "EXP1")
            store.close()

            reopened = ExperimentStore(path)
            doc = reopened.get_report("R1")
            self.assertEqual(doc["observed_statistic"], {"n": 6, "d": 1})
            self.assertEqual(doc["p_value"], {"n": 1, "d": 4})
            self.assertEqual(doc["sites"]["A"]["result_version"], "A#r1")
            self.assertTrue(reopened.verify_report("R1"))
            self.assertEqual(
                reopened.get_site_result("A#r1")["result"], {"n": 4, "d": 1})
            reopened.close()

    def test_build_report_pure_function(self):
        # The pure path needs no store at all.
        results = {
            sid: compute_site_result(hourly_site(sid, sid), self.traj)
            for sid in ("A", "B", "C", "D")
        }
        report = build_report("RP", exp1_registration(), results)
        self.assertEqual(report.observed_statistic, F(6))
        self.assertEqual(report.p_value, F(1, 4))


class OldLayersStillWorkTests(unittest.TestCase):
    """The pre-existing power query path is untouched by the extension."""

    def test_trajectory_queries_and_revision_history(self):
        store = make_store(main_traces())
        self.assertEqual(store.trajectory("A").integral(F(0), F(2)), F(8))
        store.revise_json("A", {
            "revision": "a2",
            "replace_segment": {"segment_id": "A1", "start": 1, "end": 2,
                                "power": 6},
        })
        self.assertEqual(store.trajectory("A").integral(F(0), F(2)), F(12))
        # The superseded revision remains readable for audits.
        self.assertEqual(store.trajectory_at("A", "a1").integral(F(0), F(2)), F(8))
        self.assertEqual(store.current("A").revision, "a2")


if __name__ == "__main__":
    unittest.main()
