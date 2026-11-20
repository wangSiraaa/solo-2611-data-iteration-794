"""Replay an experiment material directory into a persistent report chain.

Usage::

    python3 -m power_billing.experiment_cli materials/experiment [--db exp.db]

The CLI mirrors ``power_billing.cli``: it exists to demonstrate the chain
from versioned meters to a reviewable experiment report on the provided
materials, not to define any experiment policy.  With ``--db`` the whole
chain (site registrations, site result versions, frozen experiments,
published reports) is persisted and can be re-read and re-verified later.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .experiment import (
    SITE_AMBIGUOUS_HANDOVER,
    SITE_MISSING_EVIDENCE,
    SITE_OK,
    ExperimentRegistration,
    SiteRegistration,
)
from .experiment_store import ExperimentStore
from .power import TrajectoryStore


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _fs(frac: dict | None) -> str:
    if frac is None:
        return "null"
    n, d = frac["n"], frac["d"]
    return str(n) if d == 1 else f"{n}/{d}"


def _print_site_result(res) -> None:
    doc = res.to_json()
    if res.status == SITE_OK:
        print(f"SITE {res.result_version} ok result={_fs(doc['result'])}")
    elif res.status == SITE_MISSING_EVIDENCE:
        gaps = ", ".join(
            f"[{_fs(g['start'])},{_fs(g['end'])}) on {g['meter']}"
            for g in doc["gaps"]
        )
        print(f"SITE {res.result_version} missing_evidence gaps: {gaps}")
    elif res.status == SITE_AMBIGUOUS_HANDOVER:
        cps = ", ".join(
            f"cut={_fs(c[0])} -> {_fs(c[1])}" for c in doc["cut_points"]
        )
        print(f"SITE {res.result_version} ambiguous_handover: {cps}")


def _print_report(doc: dict) -> None:
    if doc["status"] == "ok":
        print(
            f"REPORT {doc['report_id']} ok "
            f"observed={_fs(doc['observed_statistic'])} "
            f"p={_fs(doc['p_value'])} "
            f"over {doc['assignments_total']} sign assignments"
        )
        for entry in doc["null_distribution"]:
            print(
                f"  T={_fs(entry['statistic'])} "
                f"x{entry['multiplicity']}"
            )
    else:
        print(f"REPORT {doc['report_id']} invalid:")
        for problem in doc["problems"]:
            print(f"  - {json.dumps(problem, sort_keys=True)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("materials", type=Path)
    ap.add_argument("--db", type=Path, default=None)
    args = ap.parse_args(argv)

    mdir: Path = args.materials
    trajectories = TrajectoryStore()
    store = ExperimentStore(str(args.db) if args.db is not None else ":memory:")

    ops = [
        json.loads(line)
        for line in (mdir / "operations.jsonl").read_text().splitlines()
        if line.strip()
    ]
    for op in ops:
        kind = op["op"]
        if kind == "register_power":
            data = _load(mdir / op.get("file", "power.json"))
            traces = data["traces"] if "traces" in data else [data]
            for trace in traces:
                rev = trajectories.register_json(trace)
                print(
                    f"POWER {rev.trace_id}@{rev.revision} registered "
                    f"({len(rev.segments)} segments)"
                )
        elif kind == "correct_power":
            rev = trajectories.revise_json(op["trace_id"], op["correction"])
            print(
                f"POWER {rev.trace_id} corrected to revision {rev.revision} "
                "(site results not recomputed yet)"
            )
        elif kind == "register_sites":
            data = _load(mdir / op.get("file", "sites.json"))
            for site in data["sites"]:
                reg = SiteRegistration.from_json(site)
                store.register_site(reg)
                print(f"SITE {reg.site_id}@{reg.site_version} registered")
        elif kind == "compute_site_results":
            for site_id in op.get("sites") or store.list_site_ids():
                res = store.compute_and_store_site_result(site_id, trajectories)
                _print_site_result(res)
        elif kind == "register_experiments":
            data = _load(mdir / op.get("file", "experiments.json"))
            for exp in data["experiments"]:
                reg = ExperimentRegistration.from_json(exp)
                store.register_experiment(reg)
                print(
                    f"EXPERIMENT {reg.experiment_id} frozen "
                    f"({len(reg.pairs)} pairs, {len(reg.sign_groups)} sign groups)"
                )
        elif kind == "publish_report":
            doc = store.publish_report(op["report_id"], op["experiment_id"])
            _print_report(doc)
        elif kind == "read_report":
            doc = store.get_report(op["report_id"])
            if doc is None:
                print(f"REPORT {op['report_id']}: not found")
            else:
                _print_report(doc)
        elif kind == "verify_report":
            store.verify_report(op["report_id"])
            print(
                f"REPORT {op['report_id']} verified: "
                "recomputation from bound versions matches"
            )
        else:
            print(f"unknown op {kind!r}", file=sys.stderr)
            return 2

    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
