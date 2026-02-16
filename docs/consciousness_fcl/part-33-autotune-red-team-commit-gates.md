# Part 33: Autotune Red-Team Commit Gates

## Purpose

Strengthen the self-upgrading loop so `tune.commit` is only allowed when a candidate overlay passes both micro-trial guardrails and adversarial robustness checks.

## Implementation

1. `autotune` red-team guard path
- File: `agent_forge/src/agent_forge/consciousness/modules/autotune.py`
- Added `_run_red_team_guard(...)` that runs a quick `ConsciousnessRedTeamCampaign` for the candidate overlay and returns:
  - `guard_ok`
  - guard failure reasons
  - structured campaign report payload
- Added emitted events:
  - `tune.red_team` for every executed guard evaluation
  - `tune.red_team_error` when campaign execution fails

2. Commit/rollback gating behavior
- `accepted` now requires all conditions:
  - micro-trial guardrails pass
  - score improves by configured delta
  - red-team guard passes (when enabled)
- `tune.rollback` reasons now preserve explicit red-team failure causes such as:
  - `red_team_unavailable`
  - `red_team_pass_ratio`
  - `red_team_robustness`

3. Runtime defaults
- File: `agent_forge/src/agent_forge/consciousness/types.py`
- Added config defaults:
  - `autotune_run_red_team`
  - `autotune_red_team_quick`
  - `autotune_red_team_max_scenarios`
  - `autotune_red_team_seed_offset`
  - `autotune_red_team_min_pass_ratio`
  - `autotune_red_team_min_robustness`
  - `autotune_red_team_require_available`
  - `autotune_red_team_persist`
  - `autotune_red_team_disable_modules`

4. Red-team campaign overlay/disable passthrough
- File: `agent_forge/src/agent_forge/consciousness/bench/red_team.py`
- Campaign supports shared:
  - `overlay`
  - `disable_modules`
- This allows autotune to evaluate exactly the candidate runtime profile under adversarial conditions.

## Guard Semantics

1. Availability guard
- If campaign execution fails and `autotune_red_team_require_available=true`, commit is blocked.

2. Threshold guard
- Candidate overlay must satisfy:
  - `pass_ratio >= autotune_red_team_min_pass_ratio`
  - `mean_robustness >= autotune_red_team_min_robustness`

3. Safety-first rollback
- Any threshold miss or unavailable campaign maps to deterministic rollback reasons and prevents overlay persistence.

## Validation

Run with `eidosian_venv` and repository-aware `PYTHONPATH`:

```sh
PYTHONPATH=lib:agent_forge/src:memory_forge/src:knowledge_forge/src:eidos_mcp/src ./eidosian_venv/bin/python -m pytest -q agent_forge/tests/test_consciousness_autotune.py
PYTHONPATH=lib:agent_forge/src:memory_forge/src:knowledge_forge/src:eidos_mcp/src ./eidosian_venv/bin/python -m pytest -q agent_forge/tests/test_consciousness_red_team.py
PYTHONPATH=lib:agent_forge/src:memory_forge/src:knowledge_forge/src:eidos_mcp/src ./eidosian_venv/bin/python -m pytest -q agent_forge/tests/test_consciousness_integrated_benchmark.py
PYTHONPATH=lib:agent_forge/src:memory_forge/src:knowledge_forge/src:eidos_mcp/src ./eidosian_venv/bin/python -m pytest -q eidos_mcp/tests/test_mcp_tool_calls_individual.py -k "consciousness_kernel_red_team or consciousness_kernel_full_benchmark"
```

## Acceptance Criteria

- `tune.commit` requires adversarial robustness checks when enabled.
- `tune.rollback` explicitly explains red-team gating failures.
- Quick red-team execution remains bounded and configurable for Termux.
