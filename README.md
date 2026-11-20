# Exact interval-power & demand-billing library

Python 3.11, standard library only (`fractions`, `sqlite3`). All arithmetic is
exact rational arithmetic; JSON floats are rejected rather than coerced.

Three layers:

| layer | module | responsibility |
|-------|--------|----------------|
| power / demand | `power_billing/interval.py`, `demand.py`, `power.py` | exact interval products, revisioned trajectories, demand search, target-charge products |
| billing ledger | `power_billing/ledger.py` | immutable posted bills, append-only entries, linear correction chain, current head |
| paired experiments | `power_billing/experiment.py`, `experiment_store.py` | site baseline/observation differences, one-handover meter changes, frozen pair randomization, exact randomization reports |

The first two layers are intentionally independent: a *target product* is a
pure function of `(contract, power trajectory)`; the ledger never recomputes
demand. A power correction changes only which target products are valid —
**it never books money.** Money moves only through an explicit `post_target`.

## Run

```bash
python3 -m power_billing.cli materials/735                    # replay the billing material ops
python3 -m power_billing.experiment_cli materials/experiment  # replay the experiment chain
python3 -m unittest discover -s tests -v                      # 66 tests
python3 materials/735/bill_reader.py bill.json                # read a demand-bill/v1 doc
```

`cli.py` prints `BILL B1 …`, `POWER corrected … (no amount booked)`, etc. The
receipts it emits are accepted by the supplied `bill_reader.py`
(e.g. `B1 complete 5:20 6 0`). `experiment_cli.py` prints `SITE A#r1 …`,
`REPORT R1 ok observed=6 p=1/4 …`; with `--db exp.db` the whole chain is
persisted and reports can be re-read and re-verified later.

## Layer 1 — exact interval power

A trajectory is a set of half-open segments `[start, end)` with a stable
`segment_id`, positive length and a non-negative rational power. Segments may
have gaps but never overlap; touching is allowed.

* `Trajectory.integral(a, b)` — exact integral of the **known** power over
  `[a,b)`. Gaps contribute nothing (never zero-filled, never extrapolated); use
  `coverage()` / `is_covered()` to distinguish "unknown" from "zero".
* `Trajectory.coverage(a, b)` — `(known_length, total_length, gaps)`, with the
  maximal uncovered sub-intervals, i.e. the gap provenance.
* `Trajectory.demand_max(a, b, W)` — the maximum average known power over a full
  window of length `W`, the earliest optimal window start, and the window's
  source segments.

### Why a non-breakpoint start is required

For a piecewise-constant left-continuous curve, the window integral
`I(s) = ∫_s^{s+W} P` is continuous and piecewise-linear in `s`, with slope
`P(s+W) − P(s)`. The slope can change only where `s` or `s+W` crosses a power
breakpoint. So `I` is linear between **events**
`{breakpoints} ∪ {breakpoint − W}` and its maximum on each linear piece is at an
endpoint. The candidate set is therefore

```
{events in [a, b−W]} ∪ {a, b−W}
```

(the last two are the feasible-segment endpoints that clip the search to
`[a, b−W]`). We evaluate every candidate exactly and take strict-greater, so
ties resolve to the earliest start. This finds the material p2 optimum
`[25,40)` with demand `20/3`, which a breakpoint-only search (`0/10/20`) misses.

### Complexity and revision strategy (chosen explicitly)

* `integral` is O(log n) via a prefix-integral array + binary search.
* `is_covered` is O(log n) via the sorted interior-gap list.
* `demand_max` is O(E · log n) = O(n log n), where `E ≈ 2n` events, plus O(n)
  for the final source decomposition. It is exact and never uses a fixed grid.
* Candidates depend only on `W`, so repeated queries at the same `W` reuse the
  same event geometry; different `W` recompute shifted events. Revisions rebuild
  the prefix array in O(n). We deliberately optimised for **revising cheaply and
  staying exact** rather than for caching per-`W` slope scans; at the stated
  scale (≤1000 segments) an O(n log n) exact search is comfortably fast (see
  `ScaleTests`, 200 evaluations well under a second).

### Revision

`TrajectoryStore.revise_json(...)` replaces segments **by identity** (every
replacement must name an existing `segment_id`), validates the whole candidate
trajectory (unique ids, positive lengths, non-overlap), and commits only on
success. A batch with any illegal result leaves the store byte-for-byte
unchanged. Every revision is retained: `trajectory_at(trace_id, revision)`
re-reads any superseded version, so published results that bind a revision can
always be audited against exactly that version.

## Layer 3 — paired energy-saving experiments

The chain is *versioned meters → site registrations → versioned site results
→ frozen experiment registration → published, re-verifiable report*
(`experiment.py` for the pure computation, `experiment_store.py` for the
SQLite persistence).

**Site results (collector side).** A `SiteRegistration` freezes per site one
or more disjoint baseline periods and one observation period (positive
lengths; boundaries may fall inside measurement segments). The site
difference is

```
(total baseline energy / total baseline duration)
    × observation duration − measured observation energy
```

in exact rationals — when the two ends have different durations, energies are
never subtracted directly and per-period mean powers are never averaged. The
result binds the site, the periods, the trajectory revisions and the rational
quantities of both ends. Any gap in a required period yields
`missing_evidence` with the uncovered sub-intervals (a measured power `0` is
data; a gap is no measurement).

**One meter change per site.** A site may switch meters at most once. When
the switch is only known to lie in the closed interval `[L, U]`, the old
meter reads before the unknown cut point and the new one from the cut point
on; baseline and observation share that single handover. The site difference
is adopted only if it is identical for **every** legal cut point — the
difference is a continuous piecewise-linear function of the cut point whose
kinks lie at meter breakpoints and period endpoints, so checking that finite
candidate set decides constancy. Every positive-length interval some legal
cut point could actually use must be measured (`[L,U]` requires *both*
meters over the required periods), else `missing_evidence` names the gap; if
readings are complete but the value still differs, two cut points with
different results are reported (`ambiguous_handover`). A trajectory
correction followed by recomputation produces a new site-result version
(`A#r1` → `A#r2`); published reports keep pointing at the versions they used.

**Reports (experimenter side).** An `ExperimentRegistration` freezes the site
pairs, each pair's left/right order, the shared sign groups with their
`same`/`reverse` wiring, and the observed sign faces. Sign groups come from
the experimenter's actual randomization scheme — they are declared, never
estimated from meter correlations. Each group independently picks `+`/`−`
with equal probability (at most twelve groups); pairs wired to one group flip
together and exactly one site per pair is treated. The statistic is the sum
over all pairs of (treated − control site difference). `build_report`
enumerates every allowed sign-face combination under the null hypothesis of
no treatment effect, tallies the statistic values with their multiplicities
(repeated values still count their sign faces) and gives the exact one-sided
probability of a statistic at least the observed one. Any frozen site
without evidence, or an observed assignment contradicting the registered
wiring, makes the report `invalid` with the reasons listed — a pair is never
dropped to continue with a different denominator.

**Persistence and review.** `ExperimentStore` keeps site registrations
(versioned, immutable), site results (sequential `site#rN` versions,
idempotent on identical content), frozen experiment registrations
(re-registration with different content is rejected) and immutable reports
with the same idempotent-replay rule as the billing ledger. Every report
binds the registration fingerprint and the exact site-result versions it
used; `verify_report(report_id)` recomputes the report from those bound
versions and compares it against the stored document.

## Layer 4 — demand-bill-v1 products

`evaluate(contract, trajectory, trace, revision)` returns a `TargetCharge`:

| `coverage` | meaning | `window` / `demand` | `charge` |
|------------|---------|---------------------|----------|
| `complete` | entire `[a,b)` covered by known segments | best window | `max(0,D−D0)×q` |
| `partial` | gaps in `[a,b)` but some full known window exists | best **known** window | `null` |
| `insufficient` | no fully-known window of length `W` | `null` | `null` |

Only `complete` is bookable. A partial result is explicitly *not* the
full-period maximum and can never become a successful (zero or non-zero) bill —
it may only be stored via `record_analysis`, which creates no bill, entry, or
head. Insufficient evidence likewise never yields a zero-charge success.

The `demand-bill/v1` document (see `bill_document`) preserves
`settlement_id`, `coverage`, `window`, `demand`, `charge` in the required
fraction-object form, and additionally carries `predecessor_id`, `difference`,
`entry_id`, `balance_after`, `sources`, `gaps`, versions and period.

## Layer 4 — the ledger

State per `contract_id` + exact `period`:

* `bills` — immutable rows: target `charge`, booked `delta`, `predecessor_id`,
  `balance_after`, full receipt JSON. One row per `settlement_id`.
* `entries` — append-only borrow/credit entries, exactly one per bill, amount =
  the bill's `delta`.
* `heads` — the current bill for each `contract_id` + `period`.

Rules enforced inside a single `BEGIN IMMEDIATE` transaction:

1. **First bill** — no `expected_previous_bill`; `delta = target`; balance from 0.
2. **Correction** — `expected_previous_bill` must equal the current head; the
   booked amount is `new_target − predecessor_target` (never the full new
   amount); the head flips. Zero difference still advances the head and adds a
   zero entry.
3. **No fork** — `UNIQUE(predecessor_id)` plus the head check means a
   predecessor can have at most one successor; of two concurrent corrections
   from the same head, at most one commits, the other gets `ConflictError` and
   must re-read.
4. **Versions** — the product's contract revision and power revision must equal
   the registered current ones, and the product's period must match the stored
   contract.
5. **Idempotency** — replaying a `settlement_id` with the same target product
   returns the stored receipt and writes nothing; a different product is
   rejected as different content.
6. **Invariant** — after every write, `net(entries) == head.target_charge`;
   `verify()` re-checks it. Any failure rolls back the whole transaction, leaving
   no delta, partial bill, or wrong head.

### Concurrency model and its cost

Serialization is by SQLite's write lock: `BEGIN IMMEDIATE` at the start of each
posting, so all head/predecessor/version checks and the writes see a consistent
snapshot and cannot interleave. The cost is that writers to the same database
serialize; readers continue concurrently (WAL mode for file databases). This
was chosen over optimistic "validate-then-commit with a version recheck" because
it is simpler to prove correct and the workload is write-light. `busy_timeout`
(default 15 s) makes a losing writer wait rather than fail spuriously; a genuine
conflict (stale head) is reported as `ConflictError`.

## Verified material scenario

| step | result |
|------|--------|
| p1, B1 | window `[5,20)`, demand `6`, target `0`, balance `0` |
| p2 (S3→9), B2 `expected_previous_bill=B1` | window `[25,40)`, demand `20/3`, target `2`, **delta +2**, balance `2` |
| replay B2 | identical receipt, balance still `2`, no new entry |
| p3 (S3→2), B3 `expected_previous_bill=B2` | demand `6`, target `0`, **delta −2**, balance `0` |
| B1/B2 after corrections | read back unchanged |
| two corrections from B2 | one commits a single −2, the other conflicts; entries `[0,+2,−2]`, not two extra −2 entries |

Edge cases covered by tests: `W` = whole period, flat-peak ties (earliest
start), head/tail truncation, gaps (partial), no known window (insufficient),
overlap rejection, all-or-nothing batch revision, zero-difference head advance,
idempotent replay, conflicting id reuse, and reopening a file database.

## Verified experiment scenario

One-hour equal-length baseline/observation; two pairs with (left baseline,
left observation; right baseline, right observation) powers `(6, 2; 2, 2)`
and `(4, 2; 2, 2)`; both pairs' left sites treated at observation:

| step | result |
|------|--------|
| EXP1: two independent sign groups | observed statistic `6`, one-sided `p = 1/4` |
| EXP2: one shared sign, same-direction wiring | observed statistic `6`, one-sided `p = 1/2` |
| correct pair-1 left observation power `2 → 6` (trace `A` `a1 → a2`, site result `A#r1 → A#r2`), re-publish on EXP1 | statistic `2`, `p = 1/2`; null distribution `±2` each with multiplicity 2 |
| R1 after the correction | still reads `6`, `p = 1/4`, bound to `A#r1`; `verify_report` reproduces it |

Demonstrations alongside (`materials/experiment`, sites `E`, `F0`, `F1`, `H`):
unequal-length baselines use the energy-weighted mean (`12/3 = 4`, not the
average of per-period means `5`); a measured power `0` gives a result
(`4`), while an unmeasured observation gives `missing_evidence` (`[1,2)`);
the handover example (baseline `[0,1)`, observation `[1,2)`, old meter
constant `2`, new constant `1`, cut point in `[0,2]`) is
`ambiguous_handover` — cut points `0` or `2` give difference `0`, cut point
`1` gives `1`, so two such cut points are reported and nothing is adopted.

## Requirement → test map

| requirement | test |
|---|---|
| exact interval integral; gaps not zero-filled | `test_prefix_integral_matches_piecewise_sum`, `test_uncovered_time_is_not_zero_filled_or_extended` |
| optimal window may start between measurements | `test_material_p1_demand_and_window_start`, `test_material_p2_demand_and_charge` |
| `W` = whole period | `test_window_equals_whole_period` |
| flat/tie → earliest start | `test_flat_peak_tie_takes_earliest_start`, `test_tie_between_two_plateaus_takes_earliest` |
| head/tail truncation | `test_truncated_head_and_tail` |
| gaps → partial, charge null | `test_gap_marks_partial_and_charge_none` |
| no known window → insufficient | `test_no_known_window_is_insufficient` |
| reject overlap; identity revision atomic | `test_overlap_rejected_and_touching_allowed`, `test_invalid_revision_leaves_store_unchanged`, `test_multi_segment_batch_all_or_nothing` |
| correction alone books nothing | `test_power_correction_alone_books_nothing` |
| full B1/B2/B3 scenario | `test_full_scenario_b1_b2_b3` |
| idempotent replay / conflicting content | `test_replay_without_predecessor_hint_returns_original`, `test_replay_with_different_content_is_rejected`, `test_conflicting_bill_id_rejected` |
| stale predecessor rejected | `test_stale_predecessor_rejected` |
| zero difference advances head, no balance | `test_zero_difference_keeps_new_head_without_growing_balance` |
| two connections, one winner | `test_only_one_concurrent_correction_wins` |
| reopen persisted database | `test_reopening_a_database_preserves_state` |
| scale (1000 seg / 200 contracts / 500 bills) | `test_1000_segments_200_contracts_500_bills` |
| site difference = weighted baseline mean × obs duration − obs energy | `test_equal_length_one_hour_results`, `test_period_boundaries_inside_segments_are_exact` |
| unequal durations: no direct energy subtraction, no averaging of means | `test_unequal_baseline_periods_use_weighted_mean`, `test_unequal_end_durations_do_not_subtract_energies` |
| power 0 is data; no measurement is missing evidence | `test_zero_power_is_a_measurement`, `test_no_measurement_is_missing_evidence`, `test_gap_inside_baseline_reported` |
| handover adopted only if constant over all cut points; two differing cut points reported | `test_spec_example_is_ambiguous`, `test_unique_when_meters_agree_inside_handover`, `test_handover_outside_required_periods`, `test_known_cut_point_is_degenerate_interval` |
| possibly-used interval unmeasured → missing evidence | `test_missing_old_reading_inside_handover`, `test_missing_new_reading_inside_handover` |
| independent signs: statistic 6, p 1/4 | `test_independent_signs_statistic_and_probability` |
| shared sign, same-direction wiring: p 1/2 | `test_shared_sign_same_direction` |
| correction → new site-result version → statistic 2, p 1/2; old report still bound to old version | `test_correction_produces_new_result_version_and_report` |
| repeated statistic values count sign-face multiplicities | `test_correction_produces_new_result_version_and_report` (null distribution `±2 ×2`) |
| reverse wiring flips the pair | `test_reverse_wiring_flips_the_pair` |
| missing evidence / wiring violation → invalid report, no dropped pairs | `test_missing_evidence_invalidates_without_dropping_pairs`, `test_observed_assignment_violating_wiring_invalidates` |
| at most twelve sign groups | `test_sign_group_limit_is_twelve` |
| frozen registrations; idempotent replay; conflicts | `test_experiment_registration_is_frozen`, `test_site_registration_versions_and_conflicts`, `test_report_replay_and_conflict`, `test_site_result_idempotent_without_correction` |
| persisted, re-verifiable chain | `test_reopening_a_database_preserves_the_chain`, `test_build_report_pure_function` |
| old power queries keep working; superseded revisions re-readable | `test_trajectory_queries_and_revision_history` |
