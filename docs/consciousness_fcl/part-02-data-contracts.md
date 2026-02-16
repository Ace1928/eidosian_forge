# Part 02: Data Contracts

## Goal

Standardize event and payload schemas used by consciousness modules.

## Event Family

- `sense.percept`
- `intero.drive`
- `world.prediction`
- `world.prediction_error`
- `policy.action`
- `policy.efference`
- `attn.candidate`
- `gw.competition`
- `gw.ignite`
- `self.agency_estimate`
- `self.boundary_estimate`
- `meta.state_estimate`
- `report.self_report`
- `perturb.inject`
- `perturb.response`
- `metrics.*`

## Workspace Payload Contract

```json
{
  "kind": "PERCEPT|DRIVE|PRED_ERR|GOAL|PLAN|SELF|META|REPORT|PERTURB|METRIC|GW_WINNER",
  "ts": "ISO-8601",
  "id": "string",
  "source_module": "string",
  "confidence": 0.0,
  "salience": 0.0,
  "content": {},
  "links": {
    "corr_id": "string",
    "parent_id": "string",
    "memory_ids": []
  }
}
```

## Acceptance Criteria

- Schema helper validates required keys and normalizes bounds.
- Malformed payloads are rejected or normalized without daemon crash.
