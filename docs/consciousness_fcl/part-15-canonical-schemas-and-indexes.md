# Part 15: Canonical Schemas and Event Indexes

Status: core EventIndex and runtime module migrations delivered in Stage H PR-H1/PR-H2; benchmark path migration pending.

## Goal

Introduce a uniform schema and cached index layer so causal queries, metrics, and trial analysis remain fast and correct as event volume grows.

## Core Deliverables

1. Canonical event/data schema reference for consciousness event types.
2. Beat-local `EventIndex` built once per tick and reused by modules.
3. Indexed lookups for recent causal graph construction.

## Event Type Coverage

Mandatory schema sections for:

- `attn.candidate`
- `gw.competition`
- `gw.reaction_trace`
- `gw.ignite`
- `policy.action`
- `policy.efference`
- `self.agency_estimate`
- `report.self_report`
- `perturb.inject`
- `bench.trial_result`

Each schema includes:

- required top-level keys
- required `links` keys
- semantic constraints (ranges, enum set)
- scoring relevance flag

## Target Files

- `agent_forge/src/agent_forge/consciousness/index.py` (new)
- `agent_forge/src/agent_forge/consciousness/types.py`
- `agent_forge/src/agent_forge/consciousness/schemas.py` (new)

## EventIndex Definition

```python
@dataclass
class EventIndex:
    latest_by_type: dict[str, dict]
    latest_by_module: dict[str, dict]
    broadcasts_by_kind: dict[str, list[dict]]
    by_corr_id: dict[str, list[dict]]
    children_by_parent: dict[str, list[dict]]
    candidates_by_id: dict[str, dict]
    winners_by_candidate_id: dict[str, dict]
    references_by_candidate_id: dict[str, list[dict]]
```

## TickContext Integration

Add cached property:

```python
@property
def index(self) -> EventIndex:
    ...
```

Add helpers:

- `events_by_corr_id(corr_id)`
- `children(parent_id)`
- `candidate(candidate_id)`
- `winner_for_candidate(candidate_id)`

## Performance Plan

- one-pass index build over `recent_events + emitted_events`
- no repeated reverse scans for hot modules
- module lookup path migrated incrementally

## Migration Sequence

1. Build index utilities and tests.
2. Switch `workspace_competition` reaction tracing to index helpers.
3. Switch `report` grounding checks to index helpers.
4. Switch benchmark/trial extraction to index-based traversal.

## Test Plan

### Unit

- index maps created correctly for mixed event streams
- parent-child lookup works for nested linkage
- candidate/winner mappings resolve deterministically

### Performance

- benchmark synthetic stream at 10k, 50k, 100k events
- assert index build + query remains within target envelope

## Acceptance Criteria

1. No module performing repeated full-window scans in the critical path after migration.
2. Reaction trace and report grounding use index-based lookups.
3. Trial extraction latency reduced versus baseline.

## Rollback Safety

- retain old scan methods behind fallback wrappers until tests converge
- toggle index usage by config: `consciousness_use_event_index`
