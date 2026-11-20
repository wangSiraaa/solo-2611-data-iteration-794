"""Public facade tying power trajectory, demand engine, and ledger together.

Typical use::

    store = TrajectoryStore()
    store.register_json(power_json)
    contract = Contract.from_json(contract_json)

    ledger = Ledger("billing.db")
    ledger.register_contract(contract)
    ledger.set_current_power(trace_id, revision)

    target = evaluate(contract, store.trajectory(trace_id), trace_id, revision)
    receipt = ledger.post_target("B1", target)

A power correction is two independent steps, on purpose:

1. ``store.revise_json(...)`` builds the new (validated) trajectory;
2. ``ledger.set_current_power(...)`` marks it current -- **booking nothing**.

Only an explicit ``post_target`` against the still-current versions moves money.

The experiment chain reuses the same revisioned trajectories::

    exp = ExperimentStore("experiments.db")
    exp.register_site(SiteRegistration.from_json(site_json))
    result = exp.compute_and_store_site_result("A", store)   # binds revisions
    exp.register_experiment(ExperimentRegistration.from_json(exp_json))
    report = exp.publish_report("R1", "EXP1")                # frozen inputs only
    exp.verify_report("R1")                                  # recompute & compare

A trajectory correction changes which site results are current; recomputing
stores a new ``site#rN`` version, and already published reports keep pointing
at the versions they used.
"""

from __future__ import annotations

from fractions import Fraction

from .demand import (
    COVERAGE_COMPLETE,
    COVERAGE_INSUFFICIENT,
    COVERAGE_PARTIAL,
    SCHEMA,
    Contract,
    TargetCharge,
    evaluate,
)
from .errors import (
    ConflictError,
    ContractError,
    ExperimentError,
    InsufficientEvidence,
    NotFoundError,
    PowerBillingError,
    RevisionError,
    SegmentationError,
    ValidationError,
)
from .experiment import (
    MAX_SIGN_GROUPS,
    REPORT_INVALID,
    REPORT_OK,
    SITE_AMBIGUOUS_HANDOVER,
    SITE_MISSING_EVIDENCE,
    SITE_OK,
    ExperimentRegistration,
    ExperimentReport,
    PairRegistration,
    SiteRegistration,
    SiteResult,
    build_report,
    compute_site_result,
    derived_treated,
    handover_result_at,
)
from .experiment_store import ExperimentStore
from .interval import Segment, Trajectory, period_charge
from .ledger import Ledger, bill_document
from .power import PowerRevision, TrajectoryStore

__all__ = [
    "Segment",
    "Trajectory",
    "Contract",
    "TargetCharge",
    "PowerRevision",
    "TrajectoryStore",
    "Ledger",
    "evaluate",
    "period_charge",
    "bill_document",
    "SCHEMA",
    "COVERAGE_COMPLETE",
    "COVERAGE_PARTIAL",
    "COVERAGE_INSUFFICIENT",
    "Fraction",
    "PowerBillingError",
    "ValidationError",
    "SegmentationError",
    "ContractError",
    "RevisionError",
    "InsufficientEvidence",
    "ConflictError",
    "NotFoundError",
    "ExperimentError",
    "SiteRegistration",
    "SiteResult",
    "PairRegistration",
    "ExperimentRegistration",
    "ExperimentReport",
    "ExperimentStore",
    "compute_site_result",
    "handover_result_at",
    "derived_treated",
    "build_report",
    "MAX_SIGN_GROUPS",
    "SITE_OK",
    "SITE_MISSING_EVIDENCE",
    "SITE_AMBIGUOUS_HANDOVER",
    "REPORT_OK",
    "REPORT_INVALID",
]
