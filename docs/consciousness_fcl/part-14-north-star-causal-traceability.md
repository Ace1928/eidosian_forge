# Part 14: North Star - Causal Traceability by ID

## Goal

Make every runtime claim reconstructable from IDs and links, not heuristics.

## Problem Statement

Current runtime carries `corr_id`, `parent_id`, and payload links in many places, but linkage is not guaranteed by contract. This weakens:

- winner-to-reaction causality analysis
- report grounding validation
- benchmark reproducibility
- perturbation impact attribution

## Design Rule

A consciousness event is only considered valid for scoring if it carries traceable lineage.

## Required Link Contract

All workspace payloads must contain:

```json
{
  "links": {
    "corr_id": "string",
    "parent_id": "string",
    "memory_ids": ["string"],
    "winner_candidate_id": "string|empty",
    "candidate_id": "string|empty"
  }
}
```

All emitted events that can influence workspace competition must include at least one of:

- `corr_id`
- `parent_id`
- payload `links.candidate_id`
- payload `links.winner_candidate_id`

## Target Files

- `agent_forge/src/agent_forge/consciousness/types.py`
- `agent_forge/src/agent_forge/consciousness/linking.py` (new)
- `agent_forge/src/agent_forge/consciousness/modules/*` (call-site hardening)

## Planned API

```python
def new_corr_id(self, seed: str | None = None) -> str: ...
def link(
    self,
    *,
    parent_id: str | None,
    corr_id: str | None,
    candidate_id: str | None,
    winner_candidate_id: str | None,
    memory_ids: list[str] | None = None,
) -> dict[str, object]: ...
```

## Enforcement Strategy

1. `normalize_workspace_payload` injects missing keys into `links`.
2. `TickContext.broadcast` patches missing link fields prior to write.
3. Module templates use centralized link helper.
4. Add strict mode config for development:
- `consciousness_require_links=true` raises module warnings or test failures when missing.

## Validation

### Unit tests

- missing `links` is normalized with full key set.
- missing `corr_id` creates deterministic new correlation ID.
- `winner_candidate_id` survives normalization and broadcast round trip.

### Integration tests

- attention -> competition -> working_set -> policy -> report thread has stable linkage chain.
- benchmark extraction can retrieve all descendants for a winner by `corr_id`.

## Acceptance Criteria

1. 100% of scored events in benchmark windows have canonical links.
2. Ignition trace logic can operate without time-window fallback for linked events.
3. Report evidence references resolve to actual event IDs.

## Risks

- Link inflation can add noisy IDs if generation is not constrained.
- Backward compatibility with older events without links requires fallback path.

## Mitigation

- preserve optional fallback for historical logs
- enforce strict mode only in tests and CI initially
