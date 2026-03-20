# Code Forge Archive Lifecycle

## Purpose

Define the operational contract for ingesting `archive_forge` into Code Forge in two explicit modes:

- `ingest_and_keep`: ingest archive content into the Code Forge library and keep the source repository in place.
- `ingest_and_remove`: ingest archive content into the Code Forge library, verify reversible reconstruction/parity, and then retire the source repository by moving it into a reversible retirement store.

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
  - Code Forge file records exist for the repo paths
  - reconstruction parity passes
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
- recovery path is explicit and scriptable

## Current Weaknesses

- planner refresh still has to walk the archive tree; it is lighter than full hashing, but still proportional to archive size
- degraded knowledge sync preserves capture progress but still logs noisy connection errors while embedding services are intentionally paused
- Atlas now exposes archive plan/lifecycle state and can trigger cached status refreshes and bounded waves, but repo-mode editing/retirement review still need deeper operator UX
- removal from raw archive storage is reversible retirement, not final garbage collection
- deleted files inside a kept repo are currently preserved as historical library evidence rather than purged from Code Forge

## Next Integration Work

- expand Atlas from current archive plan/lifecycle actions into full repo-mode editing, retirement dry-runs, and restore flows
- feed archive plan/lifecycle runtime evidence into proof/bundle scoring once live ingestion waves are producing reversible provenance at scale
- add proof/bundle coverage for archive lifecycle readiness and retirement evidence
- add cleaner low-load/no-embedding logging modes so intentional degraded sync does not flood operator logs
- add explicit repository reconstruction benchmarks for retired repos
- add a final garbage-collection layer only after retirement manifests are aged, reviewed, and backed up
