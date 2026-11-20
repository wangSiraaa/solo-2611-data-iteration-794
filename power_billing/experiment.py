"""Paired energy-saving experiments on versioned meter trajectories.

This module adds the experiment chain on top of the revisioned power
trajectories (``power_billing.power`` / ``interval``):

1.  **Collector registrations.**  Per site, the collector registers one or
    more disjoint baseline periods and one observation period
    (:class:`SiteRegistration`).  All periods have positive length; boundaries
    may fall inside measurement segments.  A site reads from either one meter
    trajectory or -- at most one meter change per site -- an *old* and a
    *new* trajectory with a handover interval ``[L, U]``: before the unknown
    cut point the old meter reads, from the cut point on the new one.  The
    baseline and the observation share that single handover.

2.  **Site results** (:func:`compute_site_result`).  The site difference is

        (total baseline energy / total baseline duration)
            * observation duration  -  measured observation energy

    in exact rationals.  The two ends may have different durations: energies
    are never subtracted directly, and per-period mean powers are never
    averaged -- the totals are divided first, then scaled.  The result binds
    the site, the periods, the trajectory revisions and the rational
    quantities of both ends.  Any gap in a required period yields
    ``missing_evidence`` with the uncovered sub-intervals (a measured power
    of 0 is *data*, a gap is *no measurement*).  A handover result is
    adopted only when it is identical for every legal cut point in
    ``[L, U]``; otherwise two cut points with different results are
    reported.

3.  **Experimenter registration.**  The experimenter pre-freezes the site
    pairs, the left/right order of each pair, the shared sign groups with
    their wiring (``same`` / ``reverse``) and the observed sign faces
    (:class:`ExperimentRegistration`).  Sign groups come from the
    experimenter's actual randomization scheme; they are declared, never
    estimated from meter correlations.  Each group independently picks
    heads/tails (``+`` / ``-``) with equal probability; pairs wired to one
    group flip together and exactly one site per pair is treated.

4.  **Reports** (:func:`build_report`).  The statistic is the sum over all
    pairs of (treated site result - control site result).  Under the null
    hypothesis of no treatment effect the site results are fixed, so the
    report enumerates every allowed sign-face combination, tallies the
    statistic values with their multiplicities (repeated values still count
    their sign faces) and gives the exact one-sided probability of a
    statistic at least the observed one.  Any frozen site without evidence,
    or an observed assignment contradicting the registered wiring, makes the
    report ``invalid`` with the reasons listed -- pairs are never dropped to
    keep a denominator going.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import Iterable, Mapping

from .errors import ExperimentError, ValidationError
from .interval import Trajectory, frac_from_json, frac_to_json
from .numbers import parse_frac
from .power import TrajectoryStore

SCHEMA_SITE_REGISTRATION = "site-registration/v1"
SCHEMA_SITE_RESULT = "site-result/v1"
SCHEMA_EXPERIMENT_REGISTRATION = "experiment-registration/v1"
SCHEMA_EXPERIMENT_REPORT = "experiment-report/v1"

#: Hard limit on the number of independent sign groups per experiment.
MAX_SIGN_GROUPS = 12

SITE_OK = "ok"
SITE_MISSING_EVIDENCE = "missing_evidence"
SITE_AMBIGUOUS_HANDOVER = "ambiguous_handover"

REPORT_OK = "ok"
REPORT_INVALID = "invalid"

WIRING_SAME = "same"
WIRING_REVERSE = "reverse"
FACE_HEADS = "+"
FACE_TAILS = "-"
TREATED_LEFT = "left"
TREATED_RIGHT = "right"

ZERO = Fraction(0)


# --------------------------------------------------------------------------- #
# Canonical JSON / fingerprints
# --------------------------------------------------------------------------- #
def canonical_json(node) -> str:
    return json.dumps(node, sort_keys=True, separators=(",", ":"))


def content_hash(node) -> str:
    """Short content binding used to pin a registration inside a report."""
    return hashlib.sha256(canonical_json(node).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Period helpers
# --------------------------------------------------------------------------- #
def _parse_period(node, field: str) -> tuple[Fraction, Fraction]:
    if not isinstance(node, (list, tuple)) or len(node) != 2:
        raise ExperimentError(f"{field}: must be a [start, end] pair")
    try:
        a = parse_frac(node[0], f"{field} start")
        b = parse_frac(node[1], f"{field} end")
    except ValidationError as exc:
        raise ExperimentError(str(exc)) from exc
    return (a, b)


def _period_json(period: tuple[Fraction, Fraction]) -> list:
    return [frac_to_json(period[0]), frac_to_json(period[1])]


def _period_from_json(node) -> tuple[Fraction, Fraction]:
    return (frac_from_json(node[0]), frac_from_json(node[1]))


def _gaps_json(gaps: Iterable[dict]) -> list:
    return [
        {
            "start": frac_to_json(g["start"]),
            "end": frac_to_json(g["end"]),
            "meter": g["meter"],
            "period": _period_json(g["period"]),
        }
        for g in gaps
    ]


def _gaps_from_json(node) -> tuple:
    return tuple(
        {
            "start": frac_from_json(g["start"]),
            "end": frac_from_json(g["end"]),
            "meter": g["meter"],
            "period": _period_from_json(g["period"]),
        }
        for g in node
    )


def _cut_points_json(cut_points: Iterable[tuple[Fraction, Fraction]]) -> list:
    return [[frac_to_json(t), frac_to_json(v)] for t, v in cut_points]


def _cut_points_from_json(node) -> tuple:
    return tuple((frac_from_json(t), frac_from_json(v)) for t, v in node)


# --------------------------------------------------------------------------- #
# Site registration (collector side)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class SiteRegistration:
    """A site's frozen measurement plan.

    Exactly one of two shapes is used:

    * single meter: ``trace_id`` set, no handover;
    * one meter change: ``old_trace_id`` + ``new_trace_id`` + ``handover``
      ``[L, U]`` (the unknown cut point lies in the closed interval).
    """

    site_id: str
    site_version: str
    baselines: tuple[tuple[Fraction, Fraction], ...]
    observation: tuple[Fraction, Fraction]
    trace_id: str | None = None
    old_trace_id: str | None = None
    new_trace_id: str | None = None
    handover: tuple[Fraction, Fraction] | None = None

    @property
    def is_handover(self) -> bool:
        return self.handover is not None

    @property
    def required_periods(self) -> tuple[tuple[Fraction, Fraction], ...]:
        return self.baselines + (self.observation,)

    def validate(self) -> None:
        if not self.site_id:
            raise ExperimentError("site_id is required")
        if not self.site_version:
            raise ExperimentError("site_version is required")
        if not self.baselines:
            raise ExperimentError("at least one baseline period is required")
        periods = list(self.baselines) + [self.observation]
        for a, b in periods:
            if not (a < b):
                raise ExperimentError("every period must have positive length")
        ordered = sorted(periods)
        for (_, b0), (a1, _) in zip(ordered, ordered[1:]):
            if a1 < b0:
                raise ExperimentError(
                    "baseline and observation periods must be disjoint"
                )
        if self.is_handover:
            if self.trace_id is not None:
                raise ExperimentError(
                    "trace_id cannot be combined with a handover"
                )
            if not self.old_trace_id or not self.new_trace_id:
                raise ExperimentError("a handover needs old/new trace ids")
            low, up = self.handover
            if low > up:
                raise ExperimentError("handover interval must satisfy L <= U")
        else:
            if not self.trace_id:
                raise ExperimentError("trace_id is required")
            if self.old_trace_id or self.new_trace_id:
                raise ExperimentError(
                    "old/new trace ids require a handover interval"
                )

    @classmethod
    def from_json(cls, obj: dict) -> "SiteRegistration":
        try:
            baselines = tuple(
                _parse_period(p, "baseline") for p in obj["baselines"]
            )
            observation = _parse_period(obj["observation"], "observation")
            site_id = str(obj["site_id"])
        except KeyError as exc:
            raise ExperimentError(f"site registration missing {exc}") from exc
        handover = obj.get("handover")
        reg = cls(
            site_id=site_id,
            site_version=str(obj.get("site_version", "v1")),
            baselines=baselines,
            observation=observation,
            trace_id=obj.get("trace_id"),
            old_trace_id=obj.get("old_trace_id"),
            new_trace_id=obj.get("new_trace_id"),
            handover=(
                _parse_period(handover, "handover") if handover is not None else None
            ),
        )
        reg.validate()
        return reg

    def to_json(self) -> dict:
        return {
            "schema": SCHEMA_SITE_REGISTRATION,
            "site_id": self.site_id,
            "site_version": self.site_version,
            "baselines": [_period_json(p) for p in self.baselines],
            "observation": _period_json(self.observation),
            "trace_id": self.trace_id,
            "old_trace_id": self.old_trace_id,
            "new_trace_id": self.new_trace_id,
            "handover": (
                _period_json(self.handover) if self.handover is not None else None
            ),
        }


# --------------------------------------------------------------------------- #
# Site results
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class SiteResult:
    """The outcome of evaluating a site registration against a store.

    ``status`` is one of ``ok`` / ``missing_evidence`` /
    ``ambiguous_handover``.  Only an ``ok`` result carries an adopted
    ``result`` value.  ``meters`` is the JSON binding of the trajectory
    revisions the evaluation used.
    """

    site_id: str
    site_version: str
    status: str
    meters: dict
    baselines: tuple[tuple[Fraction, Fraction], ...]
    observation: tuple[Fraction, Fraction]
    result_version: str | None = None
    baseline_energy: Fraction | None = None
    baseline_duration: Fraction | None = None
    baseline_mean_power: Fraction | None = None
    observation_energy: Fraction | None = None
    observation_duration: Fraction | None = None
    expected_observation: Fraction | None = None
    result: Fraction | None = None
    gaps: tuple = ()
    cut_points: tuple = ()

    def to_json(self) -> dict:
        return {
            "schema": SCHEMA_SITE_RESULT,
            "site_id": self.site_id,
            "site_version": self.site_version,
            "result_version": self.result_version,
            "status": self.status,
            "baselines": [_period_json(p) for p in self.baselines],
            "observation": _period_json(self.observation),
            "meters": self.meters,
            "baseline_energy": frac_to_json(self.baseline_energy),
            "baseline_duration": frac_to_json(self.baseline_duration),
            "baseline_mean_power": frac_to_json(self.baseline_mean_power),
            "observation_energy": frac_to_json(self.observation_energy),
            "observation_duration": frac_to_json(self.observation_duration),
            "expected_observation": frac_to_json(self.expected_observation),
            "result": frac_to_json(self.result),
            "gaps": _gaps_json(self.gaps),
            "cut_points": _cut_points_json(self.cut_points),
        }

    @classmethod
    def from_json(cls, obj: dict) -> "SiteResult":
        return cls(
            site_id=obj["site_id"],
            site_version=obj["site_version"],
            result_version=obj.get("result_version"),
            status=obj["status"],
            meters=obj["meters"],
            baselines=tuple(_period_from_json(p) for p in obj["baselines"]),
            observation=_period_from_json(obj["observation"]),
            baseline_energy=frac_from_json(obj.get("baseline_energy")),
            baseline_duration=frac_from_json(obj.get("baseline_duration")),
            baseline_mean_power=frac_from_json(obj.get("baseline_mean_power")),
            observation_energy=frac_from_json(obj.get("observation_energy")),
            observation_duration=frac_from_json(obj.get("observation_duration")),
            expected_observation=frac_from_json(obj.get("expected_observation")),
            result=frac_from_json(obj.get("result")),
            gaps=_gaps_from_json(obj.get("gaps", [])),
            cut_points=_cut_points_from_json(obj.get("cut_points", [])),
        )


def _combined_energy(old: Trajectory, new: Trajectory,
                     a: Fraction, b: Fraction, t: Fraction) -> Fraction:
    """Energy over ``[a, b)`` when the meter switches at ``t``.

    The old meter reads before ``t``, the new one from ``t`` on.
    """
    energy = ZERO
    old_end = min(b, t)
    if a < old_end:
        energy += old.integral(a, old_end)
    new_start = max(a, t)
    if new_start < b:
        energy += new.integral(new_start, b)
    return energy


def handover_result_at(reg: SiteRegistration, old: Trajectory, new: Trajectory,
                       t: Fraction) -> Fraction:
    """The site difference for one concrete cut point ``t`` (audit helper)."""
    base_energy = sum(
        (_combined_energy(old, new, a, b, t) for a, b in reg.baselines), ZERO
    )
    base_duration = sum((b - a for a, b in reg.baselines), ZERO)
    obs_a, obs_b = reg.observation
    obs_energy = _combined_energy(old, new, obs_a, obs_b, t)
    obs_duration = obs_b - obs_a
    return (base_energy / base_duration) * obs_duration - obs_energy


def _ends_at(reg: SiteRegistration, old: Trajectory, new: Trajectory,
             t: Fraction) -> dict:
    """Baseline/observation quantities for a concrete cut point ``t``."""
    base_energy = sum(
        (_combined_energy(old, new, a, b, t) for a, b in reg.baselines), ZERO
    )
    base_duration = sum((b - a for a, b in reg.baselines), ZERO)
    obs_a, obs_b = reg.observation
    obs_energy = _combined_energy(old, new, obs_a, obs_b, t)
    obs_duration = obs_b - obs_a
    mean = base_energy / base_duration
    return {
        "baseline_energy": base_energy,
        "baseline_duration": base_duration,
        "baseline_mean_power": mean,
        "observation_energy": obs_energy,
        "observation_duration": obs_duration,
        "expected_observation": mean * obs_duration,
        "result": mean * obs_duration - obs_energy,
    }


def compute_site_result(reg: SiteRegistration,
                        store: TrajectoryStore) -> SiteResult:
    """Evaluate a site registration against the current trajectory revisions.

    The returned result binds the revisions it used; a later trajectory
    correction simply yields a different (new) result when recomputed.
    """
    reg.validate()
    if reg.is_handover:
        return _compute_handover(reg, store)
    rev = store.current(reg.trace_id)
    traj = rev.trajectory()
    meters = {"kind": "single", "trace_id": rev.trace_id, "revision": rev.revision}
    gaps = []
    for a, b in reg.required_periods:
        for g0, g1 in traj.coverage(a, b)[2]:
            gaps.append(
                {"start": g0, "end": g1, "meter": rev.trace_id, "period": (a, b)}
            )
    if gaps:
        return SiteResult(
            site_id=reg.site_id, site_version=reg.site_version,
            status=SITE_MISSING_EVIDENCE, meters=meters,
            baselines=reg.baselines, observation=reg.observation, gaps=tuple(gaps),
        )
    base_energy = sum((traj.integral(a, b) for a, b in reg.baselines), ZERO)
    base_duration = sum((b - a for a, b in reg.baselines), ZERO)
    obs_a, obs_b = reg.observation
    obs_energy = traj.integral(obs_a, obs_b)
    obs_duration = obs_b - obs_a
    mean = base_energy / base_duration
    expected = mean * obs_duration
    return SiteResult(
        site_id=reg.site_id, site_version=reg.site_version,
        status=SITE_OK, meters=meters,
        baselines=reg.baselines, observation=reg.observation,
        baseline_energy=base_energy, baseline_duration=base_duration,
        baseline_mean_power=mean,
        observation_energy=obs_energy, observation_duration=obs_duration,
        expected_observation=expected, result=expected - obs_energy,
    )


def _compute_handover(reg: SiteRegistration,
                      store: TrajectoryStore) -> SiteResult:
    old_rev = store.current(reg.old_trace_id)
    new_rev = store.current(reg.new_trace_id)
    old, new = old_rev.trajectory(), new_rev.trajectory()
    low, up = reg.handover
    meters = {
        "kind": "handover",
        "old": {"trace_id": old_rev.trace_id, "revision": old_rev.revision},
        "new": {"trace_id": new_rev.trace_id, "revision": new_rev.revision},
        "interval": [frac_to_json(low), frac_to_json(up)],
    }

    # Coverage: every positive-length interval that some legal cut point
    # could actually use must be measured.  The old meter may be used
    # anywhere before ``up``, the new one anywhere from ``low`` on; inside
    # [low, up] both meters must therefore cover the required periods.
    gaps = []
    for a, b in reg.required_periods:
        old_end = min(b, up)
        if a < old_end:
            for g0, g1 in old.coverage(a, old_end)[2]:
                gaps.append({"start": g0, "end": g1,
                             "meter": old_rev.trace_id, "period": (a, b)})
        new_start = max(a, low)
        if new_start < b:
            for g0, g1 in new.coverage(new_start, b)[2]:
                gaps.append({"start": g0, "end": g1,
                             "meter": new_rev.trace_id, "period": (a, b)})
    if gaps:
        return SiteResult(
            site_id=reg.site_id, site_version=reg.site_version,
            status=SITE_MISSING_EVIDENCE, meters=meters,
            baselines=reg.baselines, observation=reg.observation, gaps=tuple(gaps),
        )

    # The site difference is a continuous piecewise-linear function of the
    # cut point whose kinks lie at meter breakpoints and period endpoints,
    # so checking that finite candidate set decides constancy on [low, up].
    candidates = {low, up}
    for t in old.breaks:
        if low <= t <= up:
            candidates.add(t)
    for t in new.breaks:
        if low <= t <= up:
            candidates.add(t)
    for a, b in reg.required_periods:
        if low <= a <= up:
            candidates.add(a)
        if low <= b <= up:
            candidates.add(b)

    values = {t: handover_result_at(reg, old, new, t) for t in sorted(candidates)}
    distinct = set(values.values())
    if len(distinct) == 1:
        # Constant difference over [low, up].  Because baseline and
        # observation are disjoint, each end's quantities are then constant
        # too, so evaluating them at ``low`` is exact.
        ends = _ends_at(reg, old, new, low)
        return SiteResult(
            site_id=reg.site_id, site_version=reg.site_version,
            status=SITE_OK, meters=meters,
            baselines=reg.baselines, observation=reg.observation, **ends,
        )
    low_result = min(distinct)
    high_result = max(distinct)
    t_low = min(t for t, v in values.items() if v == low_result)
    t_high = min(t for t, v in values.items() if v == high_result)
    return SiteResult(
        site_id=reg.site_id, site_version=reg.site_version,
        status=SITE_AMBIGUOUS_HANDOVER, meters=meters,
        baselines=reg.baselines, observation=reg.observation,
        cut_points=((t_low, low_result), (t_high, high_result)),
    )


# --------------------------------------------------------------------------- #
# Experiment registration (experimenter side)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class PairRegistration:
    pair_id: str
    left: str
    right: str
    group: str
    wiring: str  # WIRING_SAME | WIRING_REVERSE
    observed_treated: str  # TREATED_LEFT | TREATED_RIGHT

    @classmethod
    def from_json(cls, obj: dict) -> "PairRegistration":
        try:
            return cls(
                pair_id=str(obj["pair_id"]),
                left=str(obj["left"]),
                right=str(obj["right"]),
                group=str(obj["group"]),
                wiring=str(obj["wiring"]),
                observed_treated=str(obj["observed_treated"]),
            )
        except KeyError as exc:
            raise ExperimentError(f"pair registration missing {exc}") from exc

    def to_json(self) -> dict:
        return {
            "pair_id": self.pair_id,
            "left": self.left,
            "right": self.right,
            "group": self.group,
            "wiring": self.wiring,
            "observed_treated": self.observed_treated,
        }


def derived_treated(face: str, wiring: str) -> str:
    """The treated side implied by a group's face and the pair's wiring."""
    tau = (1 if face == FACE_HEADS else -1) * (1 if wiring == WIRING_SAME else -1)
    return TREATED_LEFT if tau > 0 else TREATED_RIGHT


@dataclass(frozen=True, slots=True)
class ExperimentRegistration:
    """The experimenter's frozen randomization plan and observed faces."""

    experiment_id: str
    sign_groups: tuple[str, ...]
    observed_faces: dict  # group -> "+" | "-"
    pairs: tuple[PairRegistration, ...]

    @property
    def required_sites(self) -> tuple[str, ...]:
        seen: list[str] = []
        for p in self.pairs:
            for sid in (p.left, p.right):
                if sid not in seen:
                    seen.append(sid)
        return tuple(seen)

    def validate(self) -> None:
        if not self.experiment_id:
            raise ExperimentError("experiment_id is required")
        if not (1 <= len(self.sign_groups) <= MAX_SIGN_GROUPS):
            raise ExperimentError(
                f"sign groups: need 1..{MAX_SIGN_GROUPS}, got {len(self.sign_groups)}"
            )
        if len(set(self.sign_groups)) != len(self.sign_groups):
            raise ExperimentError("duplicate sign group")
        if set(self.observed_faces) != set(self.sign_groups):
            raise ExperimentError(
                "observed faces must name exactly the registered sign groups"
            )
        for group, face in self.observed_faces.items():
            if face not in (FACE_HEADS, FACE_TAILS):
                raise ExperimentError(
                    f"sign group {group!r}: face must be '+' or '-'"
                )
        if not self.pairs:
            raise ExperimentError("at least one pair is required")
        pair_ids: set[str] = set()
        for p in self.pairs:
            if not p.pair_id or p.pair_id in pair_ids:
                raise ExperimentError(f"duplicate or empty pair_id {p.pair_id!r}")
            pair_ids.add(p.pair_id)
            if not p.left or not p.right or p.left == p.right:
                raise ExperimentError(
                    f"pair {p.pair_id!r}: two distinct sites are required"
                )
            if p.group not in self.sign_groups:
                raise ExperimentError(
                    f"pair {p.pair_id!r}: unknown sign group {p.group!r}"
                )
            if p.wiring not in (WIRING_SAME, WIRING_REVERSE):
                raise ExperimentError(
                    f"pair {p.pair_id!r}: wiring must be 'same' or 'reverse'"
                )
            if p.observed_treated not in (TREATED_LEFT, TREATED_RIGHT):
                raise ExperimentError(
                    f"pair {p.pair_id!r}: observed_treated must be 'left' or 'right'"
                )

    @classmethod
    def from_json(cls, obj: dict) -> "ExperimentRegistration":
        try:
            reg = cls(
                experiment_id=str(obj["experiment_id"]),
                sign_groups=tuple(str(g) for g in obj["sign_groups"]),
                observed_faces={
                    str(g): str(f) for g, f in obj["observed_faces"].items()
                },
                pairs=tuple(PairRegistration.from_json(p) for p in obj["pairs"]),
            )
        except KeyError as exc:
            raise ExperimentError(f"experiment registration missing {exc}") from exc
        reg.validate()
        return reg

    def to_json(self) -> dict:
        return {
            "schema": SCHEMA_EXPERIMENT_REGISTRATION,
            "experiment_id": self.experiment_id,
            "sign_groups": list(self.sign_groups),
            "observed_faces": dict(self.observed_faces),
            "pairs": [p.to_json() for p in self.pairs],
        }


def registration_fingerprint(reg: ExperimentRegistration) -> str:
    return content_hash(reg.to_json())


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ExperimentReport:
    """The computed report for one frozen experiment registration."""

    report_id: str
    experiment_id: str
    status: str
    registration_fingerprint: str
    sign_groups: tuple[str, ...]
    observed_faces: dict
    sites: dict
    pairs: tuple
    problems: tuple
    observed_statistic: Fraction | None
    null_distribution: tuple  # ((statistic, multiplicity), ...) sorted
    assignments_total: int | None
    p_value: Fraction | None

    def to_json(self) -> dict:
        return {
            "schema": SCHEMA_EXPERIMENT_REPORT,
            "report_id": self.report_id,
            "experiment_id": self.experiment_id,
            "status": self.status,
            "registration_fingerprint": self.registration_fingerprint,
            "sign_groups": list(self.sign_groups),
            "observed_faces": dict(self.observed_faces),
            "sites": {
                sid: {
                    "site_version": b["site_version"],
                    "result_version": b["result_version"],
                    "status": b["status"],
                    "meters": b["meters"],
                    "result": frac_to_json(b["result"]),
                }
                for sid, b in self.sites.items()
            },
            "pairs": [
                {
                    "pair_id": p["pair_id"],
                    "left": p["left"],
                    "right": p["right"],
                    "group": p["group"],
                    "wiring": p["wiring"],
                    "observed_treated": p["observed_treated"],
                    "difference": frac_to_json(p["difference"]),
                    "contribution": frac_to_json(p["contribution"]),
                }
                for p in self.pairs
            ],
            "problems": [dict(p) for p in self.problems],
            "observed_statistic": frac_to_json(self.observed_statistic),
            "null_distribution": [
                {"statistic": frac_to_json(value), "multiplicity": mult}
                for value, mult in self.null_distribution
            ],
            "assignments_total": self.assignments_total,
            "p_value": frac_to_json(self.p_value),
        }


def build_report(report_id: str, reg: ExperimentRegistration,
                 results: Mapping[str, SiteResult]) -> ExperimentReport:
    """Compute the experiment report from frozen inputs.

    ``results`` maps site_id to its (already computed) :class:`SiteResult`.
    Any frozen site without an adopted result, or an observed assignment
    that contradicts the registered wiring, makes the report ``invalid``;
    the problems are listed and no statistic is produced -- a pair is never
    dropped to continue with a different denominator.
    """
    reg.validate()
    if not report_id:
        raise ExperimentError("report_id is required")
    fingerprint = registration_fingerprint(reg)

    problems: list[dict] = []
    sites_binding: dict[str, dict] = {}
    values: dict[str, Fraction] = {}
    for sid in reg.required_sites:
        res = results.get(sid)
        if res is None:
            problems.append(
                {"site_id": sid, "reason": "no site result available"}
            )
            continue
        sites_binding[sid] = {
            "site_version": res.site_version,
            "result_version": res.result_version,
            "status": res.status,
            "meters": res.meters,
            "result": res.result,
        }
        if res.status != SITE_OK:
            problem: dict = {
                "site_id": sid,
                "reason": f"site result is {res.status}",
            }
            if res.gaps:
                problem["gaps"] = _gaps_json(res.gaps)
            if res.cut_points:
                problem["cut_points"] = _cut_points_json(res.cut_points)
            problems.append(problem)
        else:
            values[sid] = res.result

    for p in reg.pairs:
        face = reg.observed_faces[p.group]
        derived = derived_treated(face, p.wiring)
        if p.observed_treated != derived:
            problems.append(
                {
                    "pair_id": p.pair_id,
                    "reason": "observed assignment violates registered wiring",
                    "group": p.group,
                    "observed_face": face,
                    "wiring": p.wiring,
                    "observed_treated": p.observed_treated,
                    "derived_treated": derived,
                }
            )

    base = dict(
        report_id=report_id,
        experiment_id=reg.experiment_id,
        registration_fingerprint=fingerprint,
        sign_groups=reg.sign_groups,
        observed_faces=dict(reg.observed_faces),
        sites=sites_binding,
    )
    if problems:
        return ExperimentReport(
            status=REPORT_INVALID, pairs=(), problems=tuple(problems),
            observed_statistic=None, null_distribution=(),
            assignments_total=None, p_value=None, **base,
        )

    diffs = {p.pair_id: values[p.left] - values[p.right] for p in reg.pairs}
    observed = sum(
        (
            diffs[p.pair_id] if p.observed_treated == TREATED_LEFT else -diffs[p.pair_id]
            for p in reg.pairs
        ),
        ZERO,
    )
    wiring_sign = {
        p.pair_id: (1 if p.wiring == WIRING_SAME else -1) for p in reg.pairs
    }
    group_pos = {g: i for i, g in enumerate(reg.sign_groups)}
    dist: dict[Fraction, int] = {}
    for signs in product((1, -1), repeat=len(reg.sign_groups)):
        total = sum(
            (
                signs[group_pos[p.group]] * wiring_sign[p.pair_id] * diffs[p.pair_id]
                for p in reg.pairs
            ),
            ZERO,
        )
        dist[total] = dist.get(total, 0) + 1
    assignments_total = 1 << len(reg.sign_groups)
    at_least = sum(mult for value, mult in dist.items() if value >= observed)

    pairs_out = tuple(
        {
            "pair_id": p.pair_id,
            "left": p.left,
            "right": p.right,
            "group": p.group,
            "wiring": p.wiring,
            "observed_treated": p.observed_treated,
            "difference": diffs[p.pair_id],
            "contribution": (
                diffs[p.pair_id]
                if p.observed_treated == TREATED_LEFT
                else -diffs[p.pair_id]
            ),
        }
        for p in reg.pairs
    )
    return ExperimentReport(
        status=REPORT_OK, pairs=pairs_out, problems=(),
        observed_statistic=observed,
        null_distribution=tuple(sorted(dist.items())),
        assignments_total=assignments_total,
        p_value=Fraction(at_least, assignments_total), **base,
    )


__all__ = [
    "MAX_SIGN_GROUPS",
    "SCHEMA_SITE_REGISTRATION",
    "SCHEMA_SITE_RESULT",
    "SCHEMA_EXPERIMENT_REGISTRATION",
    "SCHEMA_EXPERIMENT_REPORT",
    "SITE_OK",
    "SITE_MISSING_EVIDENCE",
    "SITE_AMBIGUOUS_HANDOVER",
    "REPORT_OK",
    "REPORT_INVALID",
    "WIRING_SAME",
    "WIRING_REVERSE",
    "FACE_HEADS",
    "FACE_TAILS",
    "SiteRegistration",
    "SiteResult",
    "PairRegistration",
    "ExperimentRegistration",
    "ExperimentReport",
    "canonical_json",
    "content_hash",
    "compute_site_result",
    "handover_result_at",
    "derived_treated",
    "registration_fingerprint",
    "build_report",
]
