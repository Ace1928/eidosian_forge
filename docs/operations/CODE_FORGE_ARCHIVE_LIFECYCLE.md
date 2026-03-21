# Code Forge Archive Lifecycle

## Purpose

Define the operational contract for ingesting `archive_forge` into Code Forge in two explicit modes:

- `ingest_and_keep`: ingest archive content into the Code Forge and File Forge libraries and keep the source repository in place.
- `ingest_and_remove`: ingest archive content into the unified Code Forge + File Forge library substrate, verify reversible reconstruction/parity, and then retire the source repository by moving it into a reversible retirement store.

This contract exists so archive reduction is governed, incremental, and reversible rather than ad hoc.

## Core Guarantees

- Archive planning is metadata-first and repo-scoped.
- Batches are grouped by `repo_key`, route, and category, so retirement and provenance can be reasoned about per repository.
- Completed batches remain completed unless one of their input files changes; execution now re-checks batch signatures so changed batches reopen automatically even without a full archive rehash.
- Repositories marked `ingest_and_keep` are treated as expanding library sources.
- Repositories marked `ingest_and_remove` are not retired until:
  - all repo batches are completed
  - no repo batches are failed or pending
  - matching provenance artifacts exist for each repo batch
  - planned Code Forge file records exist for repo-indexed paths
  - File Forge covers any source-tree files outside the Code Forge plan
  - unified reconstruction parity passes against the full source tree
- Retirement is reversible: repositories are moved into a retirement store, not destructively deleted.

## Primary Scripts

- Planner: `scripts/code_forge_archive_plan.py`
- Lifecycle controller: `scripts/code_forge_archive_lifecycle.py`

## Planner Contract

`code_forge_archive_plan.py` writes:

- `data/code_forge/archive_ingestion/latest/repo_index.json`
- `data/code_forge/archive_ingestion/latest/archive_ingestion_batches.json`
- `data/code_forge/archive_ingestion/latest/archive_ingestion_state.json`
- `reports/code_forge_archive_plan/latest.json`
- `reports/code_forge_archive_plan/latest.md`
- `data/runtime/code_forge_archive_plan_status.json`
- `data/runtime/code_forge_archive_plan_history.jsonl`

Planner properties:

- uses `fast_metadata_v1` planning instead of full-content hashing
- stores `repo_key` per entry and per batch
- tracks `added_paths`, `modified_paths`, `removed_paths`
- reopens only changed batches on refresh
- preserves progress for unaffected batches

## Retention Policy Contract

Policy path:

- `data/code_forge/archive_ingestion/latest/repo_retention_policy.json`

Contract:

- `default_mode`: `ingest_and_keep` unless explicitly changed
- `repos.<repo_key>.mode`: `ingest_and_keep` or `ingest_and_remove`
- `repos.<repo_key>.reason`: optional human/operator note

## Lifecycle Commands

Show repo-level status from the cached plan/state without forcing a fresh archive walk:

```sh
./eidosian_venv/bin/python scripts/code_forge_archive_lifecycle.py status
```

Force a fresh planner walk before status if you specifically need a new metadata snapshot:

```sh
./eidosian_venv/bin/python scripts/code_forge_archive_lifecycle.py status --refresh
```

Mark a repo for reversible retirement after successful ingestion:

```sh
./eidosian_venv/bin/python scripts/code_forge_archive_lifecycle.py set-mode --repo-key <repo_key> --mode ingest_and_remove
```

Keep a repo as a standing library expansion source:

```sh
./eidosian_venv/bin/python scripts/code_forge_archive_lifecycle.py set-mode --repo-key <repo_key> --mode ingest_and_keep
```

Run an ingestion wave for specific repos only:

```sh
./eidosian_venv/bin/python scripts/code_forge_archive_lifecycle.py run-wave --repo-key <repo_key>
```

Run a bounded repo wave to validate the route without opening the full archive workload:

```sh
./eidosian_venv/bin/python scripts/code_forge_archive_lifecycle.py run-wave --repo-key <repo_key> --batch-limit 5
```

Retry only failed batches for a repo after fixing or working around a service dependency:

```sh
./eidosian_venv/bin/python scripts/code_forge_archive_lifecycle.py run-wave --repo-key <repo_key> --batch-limit 5 --retry-failed
```

Retire repos that are fully ready:

```sh
./eidosian_venv/bin/python scripts/code_forge_archive_lifecycle.py retire --repo-key <repo_key>
```

Preview retirement readiness and reconstruction parity without mutating retention policy:

```sh
./eidosian_venv/bin/python scripts/code_forge_archive_lifecycle.py preview-retire --repo-key <repo_key> --assume-remove-mode
```

Targeted `status` and `preview-retire` requests are now repo-scoped internally, so checking one repo does not rebuild the full archive-wide lifecycle report first.

Preview output now distinguishes between:

- `captured_file_count`: files seen by any successful Code Forge route for that repo
- `file_record_count`: repo-indexed files reconstructable from the Code Forge library itself
- `file_forge_record_count`: files currently reversible from File Forge
- `reversible_file_count`: full source-tree files reversible from the union of Code Forge and File Forge
- `uncaptured_file_count`: repo-indexed files not yet captured by any Code Forge route
- `missing_file_count`: repo-indexed files not yet reconstructable from Code Forge
- `source_tree_unindexed_count`: source-tree files outside the Code Forge plan
- `source_tree_unindexed_reversible_count`: unindexed source-tree files currently covered by File Forge

Dry-run retirement without moving sources:

```sh
./eidosian_venv/bin/python scripts/code_forge_archive_lifecycle.py retire --repo-key <repo_key> --dry-run
```

Restore a retired repo:

```sh
./eidosian_venv/bin/python scripts/code_forge_archive_lifecycle.py restore --repo-key <repo_key>
```

## Retirement Artifacts

Retirement writes into:

- `data/code_forge/archive_ingestion/latest/retirements/<timestamp>/retirement_manifest.json`
- `data/code_forge/archive_ingestion/latest/retirements/latest.json`
- `data/code_forge/archive_ingestion/latest/retirements/history.jsonl`

Per retired repo, the run records:

- source root
- retired root
- reconstruction manifest path
- parity report path
- restore command

## Strengths

- practical on large archives because planning is metadata-first
- repo-scoped batches make keep/remove decisions tractable
- incremental refresh avoids re-opening unaffected work
- code and metadata ingestion now degrade gracefully when embedding-backed sync is unavailable, so low-load library capture can continue and record integration errors instead of hard-failing the whole wave
- retirement is gated by provenance and parity, not just completion counters
- lifecycle previews now surface concrete blockers and sample paths instead of only a boolean readiness flag
- recovery path is explicit and scriptable

## Current Weaknesses

- planner refresh still has to walk the archive tree; it is lighter than full hashing, but still proportional to archive size
- degraded knowledge sync preserves capture progress but still logs noisy connection errors while embedding services are intentionally paused
- Atlas now exposes archive plan/lifecycle state, repo-mode editing, bounded waves, preview-retire actions, dry-run retirement actions, and restore actions
- removal from raw archive storage is reversible retirement, not final garbage collection
- deleted files inside a kept repo are currently preserved as historical library evidence rather than purged from Code Forge

## Next Integration Work

- continue widening Atlas from archive plan/lifecycle actions into richer retirement review, provenance drill-down, and restore validation flows
- feed archive plan/lifecycle runtime evidence into proof/bundle scoring once live ingestion waves are producing reversible provenance at scale
- keep tightening reconstructable coverage so non-code/document/config routes can eventually participate in reversible retirement guarantees, not only capture guarantees
- add proof/bundle coverage for archive lifecycle readiness and retirement evidence
- add cleaner low-load/no-embedding logging modes so intentional degraded sync does not flood operator logs
- add explicit repository reconstruction benchmarks for retired repos
- add a final garbage-collection layer only after retirement manifests are aged, reviewed, and backed up

## Unified Reconstruction Notes

- `run-wave` now performs repo-scoped File Forge indexing after Code Forge batch execution.
- Code Forge remains the semantic/code abstraction layer and provenance surface.
- File Forge is the byte-faithful reversible layer for documents, configs, workspace files, and any other source-tree artifacts outside the Code Forge plan.
- Preview/retire/restore operations now reconstruct from the union of both libraries, with File Forge allowed to overwrite stale reconstructed bytes so parity is exact.
- Live validation on `archive_forge/eidos_v1_concept` now reaches `retirement_ready=true` and `parity_pass=true` in preview mode under this unified contract.


## Prune Retired Trees

- `prune-retired` verifies that a retired repo can be reconstructed from Code Forge + File Forge before deleting the stored retired tree.
- This is the storage-reclaim step for `ingest_and_remove`.
- After pruning, `restore` reconstructs the repo back into `archive_forge/<repo_key>` from the forge libraries instead of depending on the stored retired tree.
- Atlas exposes the same operation through `POST /api/code-forge/archive-lifecycle/prune-retired`.
