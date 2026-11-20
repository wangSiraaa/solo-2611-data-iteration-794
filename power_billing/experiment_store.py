"""SQLite persistence for the experiment chain.

Stores, in one database:

* **site registrations** -- versioned per ``site_id``, immutable per
  ``(site_id, site_version)``; a ``current_sites`` pointer tracks the latest
  registered version;
* **site results** -- every computed :class:`SiteResult` gets a sequential
  ``{site_id}#r{n}`` version; identical content (same site, periods, meter
  revisions, quantities) is idempotent and reuses its version, so a
  trajectory correction followed by recomputation naturally yields a *new*
  version while a plain recompute does not;
* **experiment registrations** -- frozen at first registration; re-registering
  identical content is a no-op, different content under the same id is
  rejected;
* **reports** -- immutable per ``report_id`` with the same idempotent-replay
  rule as bills in the billing ledger.

A published report binds the registration fingerprint and the exact site
result versions it used, and :meth:`ExperimentStore.verify_report`
recomputes the report from those bound versions and compares it against the
stored document -- the audit ("复查") path.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import replace
from typing import Iterator, Mapping

from .errors import ConflictError, NotFoundError
from .experiment import (
    ExperimentRegistration,
    SiteRegistration,
    SiteResult,
    build_report,
    canonical_json,
    compute_site_result,
)
from .power import TrajectoryStore

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS site_registrations (
    site_id TEXT NOT NULL,
    site_version TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    doc_json TEXT NOT NULL,
    created_seq INTEGER NOT NULL,
    PRIMARY KEY (site_id, site_version)
);

CREATE TABLE IF NOT EXISTS current_sites (
    site_id TEXT PRIMARY KEY,
    site_version TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS site_results (
    result_version TEXT PRIMARY KEY,
    site_id TEXT NOT NULL,
    site_version TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    doc_json TEXT NOT NULL,
    created_seq INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS site_results_by_site
    ON site_results (site_id, created_seq);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    fingerprint TEXT NOT NULL,
    doc_json TEXT NOT NULL,
    created_seq INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS reports (
    report_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    doc_json TEXT NOT NULL,
    created_seq INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS reports_by_experiment
    ON reports (experiment_id, created_seq);
"""


class ExperimentStore:
    """A single SQLite-backed experiment-chain store."""

    def __init__(self, path: str = ":memory:", *, busy_timeout_ms: int = 15000):
        self.path = path
        self._conn = sqlite3.connect(
            path, isolation_level=None, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
        if path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.executescript(_SCHEMA_SQL)

    # -- lifecycle -------------------------------------------------------- #
    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ExperimentStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @contextmanager
    def _write_tx(self) -> Iterator[sqlite3.Connection]:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        else:
            self._conn.execute("COMMIT")

    def _next_seq(self, conn: sqlite3.Connection) -> int:
        row = conn.execute("SELECT value FROM meta WHERE key='seq'").fetchone()
        n = 0 if row is None else int(row["value"])
        n += 1
        conn.execute(
            "INSERT INTO meta(key, value) VALUES('seq', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(n),),
        )
        return n

    # ------------------------------------------------------------------ #
    # Site registrations
    # ------------------------------------------------------------------ #
    def register_site(self, reg: SiteRegistration) -> dict:
        """Freeze a site registration version (idempotent on identical content)."""
        reg.validate()
        doc = reg.to_json()
        fingerprint = canonical_json(doc)
        with self._write_tx() as conn:
            row = conn.execute(
                "SELECT fingerprint, doc_json FROM site_registrations "
                "WHERE site_id=? AND site_version=?",
                (reg.site_id, reg.site_version),
            ).fetchone()
            if row is not None:
                if row["fingerprint"] != fingerprint:
                    raise ConflictError(
                        f"site registration {reg.site_id!r}@{reg.site_version!r} "
                        "already exists with different content"
                    )
                return json.loads(row["doc_json"])
            seq = self._next_seq(conn)
            conn.execute(
                "INSERT INTO site_registrations(site_id, site_version, "
                "fingerprint, doc_json, created_seq) VALUES(?,?,?,?,?)",
                (reg.site_id, reg.site_version, fingerprint,
                 canonical_json(doc), seq),
            )
            conn.execute(
                "INSERT INTO current_sites(site_id, site_version) VALUES(?,?) "
                "ON CONFLICT(site_id) DO UPDATE SET site_version=excluded.site_version",
                (reg.site_id, reg.site_version),
            )
            return doc

    def get_site_registration(self, site_id: str,
                              site_version: str | None = None) -> dict | None:
        if site_version is None:
            row = self._conn.execute(
                "SELECT site_version FROM current_sites WHERE site_id=?",
                (site_id,),
            ).fetchone()
            if row is None:
                return None
            site_version = row["site_version"]
        row = self._conn.execute(
            "SELECT doc_json FROM site_registrations "
            "WHERE site_id=? AND site_version=?",
            (site_id, site_version),
        ).fetchone()
        return None if row is None else json.loads(row["doc_json"])

    def list_site_ids(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT site_id FROM current_sites ORDER BY site_id"
        ).fetchall()
        return [r["site_id"] for r in rows]

    # ------------------------------------------------------------------ #
    # Site results
    # ------------------------------------------------------------------ #
    def compute_and_store_site_result(
        self,
        site_id: str,
        trajectories: TrajectoryStore,
        *,
        site_version: str | None = None,
    ) -> SiteResult:
        """Compute the site result from current trajectory revisions and store it."""
        doc = self.get_site_registration(site_id, site_version)
        if doc is None:
            raise NotFoundError(f"no site registration for {site_id!r}")
        reg = SiteRegistration.from_json(doc)
        return self.put_site_result(compute_site_result(reg, trajectories))

    def put_site_result(self, result: SiteResult) -> SiteResult:
        """Store a computed result; identical content reuses its version."""
        doc = result.to_json()
        fp_doc = dict(doc)
        fp_doc.pop("result_version", None)
        fingerprint = canonical_json(fp_doc)
        with self._write_tx() as conn:
            row = conn.execute(
                "SELECT doc_json FROM site_results "
                "WHERE site_id=? AND fingerprint=?",
                (result.site_id, fingerprint),
            ).fetchone()
            if row is not None:
                return SiteResult.from_json(json.loads(row["doc_json"]))
            count = conn.execute(
                "SELECT COUNT(*) AS n FROM site_results WHERE site_id=?",
                (result.site_id,),
            ).fetchone()["n"]
            version = f"{result.site_id}#r{count + 1}"
            stored = replace(result, result_version=version)
            seq = self._next_seq(conn)
            conn.execute(
                "INSERT INTO site_results(result_version, site_id, site_version, "
                "fingerprint, doc_json, created_seq) VALUES(?,?,?,?,?,?)",
                (version, stored.site_id, stored.site_version, fingerprint,
                 canonical_json(stored.to_json()), seq),
            )
            return stored

    def get_site_result(self, result_version: str) -> dict | None:
        row = self._conn.execute(
            "SELECT doc_json FROM site_results WHERE result_version=?",
            (result_version,),
        ).fetchone()
        return None if row is None else json.loads(row["doc_json"])

    def latest_site_result(self, site_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT doc_json FROM site_results WHERE site_id=? "
            "ORDER BY created_seq DESC LIMIT 1",
            (site_id,),
        ).fetchone()
        return None if row is None else json.loads(row["doc_json"])

    def site_results(self, site_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT doc_json FROM site_results WHERE site_id=? "
            "ORDER BY created_seq",
            (site_id,),
        ).fetchall()
        return [json.loads(r["doc_json"]) for r in rows]

    # ------------------------------------------------------------------ #
    # Experiment registrations (frozen)
    # ------------------------------------------------------------------ #
    def register_experiment(self, reg: ExperimentRegistration) -> dict:
        """Freeze an experiment registration; different content is rejected."""
        reg.validate()
        doc = reg.to_json()
        fingerprint = canonical_json(doc)
        with self._write_tx() as conn:
            row = conn.execute(
                "SELECT fingerprint, doc_json FROM experiments "
                "WHERE experiment_id=?",
                (reg.experiment_id,),
            ).fetchone()
            if row is not None:
                if row["fingerprint"] != fingerprint:
                    raise ConflictError(
                        f"experiment {reg.experiment_id!r} is frozen; "
                        "re-registration with different content is rejected"
                    )
                return json.loads(row["doc_json"])
            seq = self._next_seq(conn)
            conn.execute(
                "INSERT INTO experiments(experiment_id, fingerprint, doc_json, "
                "created_seq) VALUES(?,?,?,?)",
                (reg.experiment_id, fingerprint, canonical_json(doc), seq),
            )
            return doc

    def get_experiment(self, experiment_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT doc_json FROM experiments WHERE experiment_id=?",
            (experiment_id,),
        ).fetchone()
        return None if row is None else json.loads(row["doc_json"])

    # ------------------------------------------------------------------ #
    # Reports
    # ------------------------------------------------------------------ #
    def publish_report(
        self,
        report_id: str,
        experiment_id: str,
        results: Mapping[str, SiteResult] | None = None,
    ) -> dict:
        """Build and immutably store the report for an experiment.

        ``results`` may explicitly supply the :class:`SiteResult` objects to
        use (they are stored first, so the report always binds stored
        versions); when omitted, the latest stored result of each frozen
        site is used.  Re-publishing identical content under the same id is
        an idempotent no-op; different content is rejected.
        """
        reg_doc = self.get_experiment(experiment_id)
        if reg_doc is None:
            raise NotFoundError(f"no experiment registration for {experiment_id!r}")
        reg = ExperimentRegistration.from_json(reg_doc)

        if results is None:
            gathered: dict[str, SiteResult] = {}
            for sid in reg.required_sites:
                latest = self.latest_site_result(sid)
                if latest is not None:
                    gathered[sid] = SiteResult.from_json(latest)
            results = gathered
        else:
            results = {
                sid: self.put_site_result(res) for sid, res in results.items()
            }

        report = build_report(report_id, reg, results)
        doc = report.to_json()
        fingerprint = canonical_json(doc)
        with self._write_tx() as conn:
            row = conn.execute(
                "SELECT fingerprint, doc_json FROM reports WHERE report_id=?",
                (report_id,),
            ).fetchone()
            if row is not None:
                if row["fingerprint"] != fingerprint:
                    raise ConflictError(
                        f"report_id {report_id!r} already exists with "
                        "different content"
                    )
                return json.loads(row["doc_json"])
            seq = self._next_seq(conn)
            conn.execute(
                "INSERT INTO reports(report_id, experiment_id, fingerprint, "
                "doc_json, created_seq) VALUES(?,?,?,?,?)",
                (report_id, experiment_id, fingerprint,
                 canonical_json(doc), seq),
            )
            return doc

    def get_report(self, report_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT doc_json FROM reports WHERE report_id=?", (report_id,)
        ).fetchone()
        return None if row is None else json.loads(row["doc_json"])

    def reports_for_experiment(self, experiment_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT doc_json FROM reports WHERE experiment_id=? "
            "ORDER BY created_seq",
            (experiment_id,),
        ).fetchall()
        return [json.loads(r["doc_json"]) for r in rows]

    def verify_report(self, report_id: str) -> bool:
        """Recompute a stored report from its bound versions and compare.

        Returns ``True`` when the recomputation reproduces the stored
        document exactly; raises :class:`NotFoundError` for an unknown
        report and :class:`ConflictError` when a bound input is missing or
        the recomputation differs.
        """
        stored = self.get_report(report_id)
        if stored is None:
            raise NotFoundError(f"no report {report_id!r}")
        reg_doc = self.get_experiment(stored["experiment_id"])
        if reg_doc is None:
            raise ConflictError(
                f"experiment registration {stored['experiment_id']!r} is missing"
            )
        reg = ExperimentRegistration.from_json(reg_doc)
        results: dict[str, SiteResult] = {}
        for sid, binding in stored["sites"].items():
            version = binding.get("result_version")
            if version is None:
                raise ConflictError(
                    f"report {report_id!r} binds no result version for {sid!r}"
                )
            doc = self.get_site_result(version)
            if doc is None:
                raise ConflictError(
                    f"site result {version!r} referenced by report "
                    f"{report_id!r} is missing"
                )
            results[sid] = SiteResult.from_json(doc)
        recomputed = build_report(report_id, reg, results)
        if canonical_json(recomputed.to_json()) != canonical_json(stored):
            raise ConflictError(
                f"report {report_id!r} does not reproduce from its bound versions"
            )
        return True


__all__ = ["ExperimentStore"]
