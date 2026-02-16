# Part 16: Winner-Linked Ignition Tracing

## Goal

Upgrade ignition detection from source-count proxy to winner-linked trace evidence.

## Current Limitation

Ignition can be over-counted by unrelated broadcast activity. We need linkage-aware reaction chains tied to winners.

## Deliverables

1. `gw.reaction_trace` event emitted per winner.
2. `trace_strength` metric computed from linked reactions.
3. Ignition criteria v3 based on trace evidence.

## Target Files

- `agent_forge/src/agent_forge/consciousness/modules/workspace_competition.py`
- `agent_forge/src/agent_forge/consciousness/metrics/ignition_trace.py` (new)
- `agent_forge/src/agent_forge/core/workspace.py` (optional summary extension)

## Reaction Match Rules

A reaction qualifies if one or more conditions hold:

- event `parent_id == winner_corr_id`
- event payload `links.parent_id == winner_corr_id`
- payload `links.winner_candidate_id == winner_candidate_id`
- payload `content.winner_candidate_id == winner_candidate_id`

## Trace Metric

```text
trace_strength =
  0.5 * min(1, distinct_sources/5)
+ 0.3 * min(1, reaction_count/10)
+ 0.2 * (1 - min(1, latency_ms/1500))
```

Normalized to `[0,1]`.

## Event Schema

`gw.reaction_trace` payload:

```json
{
  "winner_candidate_id": "...",
  "winner_corr_id": "...",
  "reaction_count": 7,
  "distinct_sources": 4,
  "modules_reacted": ["meta", "policy", "working_set", "report"],
  "latency_ms": 320,
  "trace_strength": 0.78,
  "window_secs": 1.5
}
```

## Ignition Criteria v3

Emit `gw.ignite` only if:

1. winner broadcast exists
2. linked reaction count >= threshold
3. linked distinct source count >= threshold
4. trace strength >= threshold

Config keys:

- `competition_trace_strength_min`
- `competition_reaction_min_count`
- `competition_reaction_min_sources`

## Validation

### Positive synthetic

- linked follow-up events produce high trace strength and ignition.

### Negative synthetic

- unrelated background broadcasts do not produce ignition.

### Ablation

- disabling report or policy lowers trace strength as expected.

## Acceptance Criteria

1. Ignition false-positive rate reduced in noise-heavy runs.
2. Trace metrics correlate with grounded reports and agency stability.
3. Benchmark captures trace-strength distributions by trial type.
