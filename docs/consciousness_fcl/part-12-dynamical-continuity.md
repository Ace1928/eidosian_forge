# Part 12: Dynamical Continuity and Emergent Runtime

## Goal

Upgrade the consciousness runtime from mostly event-window inference to persistent, multi-timescale, stateful dynamics with stronger perturbation realism and ignition causality.

## Implemented In This Pass

1. Persistent module state
- `agent_forge/src/agent_forge/consciousness/state_store.py`
- Per-module namespaces under `state/<...>/consciousness/module_state.json`
- Kernel beat counter persisted across restarts

2. Multi-rate scheduling
- `agent_forge/src/agent_forge/consciousness/kernel.py`
- `module_tick_periods` config support
- module disable list (`disable_modules`)

3. Indexed context + perturb lookup
- `agent_forge/src/agent_forge/consciousness/types.py`
- Added:
  - `latest_event`, `latest_events`
  - `latest_broadcast`, `latest_broadcasts`, `all_broadcasts`
  - `module_state`
  - `perturbations_for`

4. Stateful substrate modules
- `agent_forge/src/agent_forge/consciousness/modules/sense.py`
- `agent_forge/src/agent_forge/consciousness/modules/intero.py`
- `agent_forge/src/agent_forge/consciousness/modules/affect.py`
- `agent_forge/src/agent_forge/consciousness/modules/working_set.py`
- `agent_forge/src/agent_forge/consciousness/modules/__init__.py`
- Default kernel order now includes `sense -> intero -> affect -> world_model -> attention -> workspace_competition -> working_set -> policy -> self_model_ext -> meta -> report`.

5. GNW competition quality improvements
- `agent_forge/src/agent_forge/consciousness/modules/workspace_competition.py`
- Added:
  - `gw.reaction_trace`
  - winner-specific reaction linkage checks
  - stronger ignition gate using reaction source/count thresholds
  - inhibition-of-return cooldown / override score
  - perturb-aware competition behavior

6. Perturbation runtime activation
- `agent_forge/src/agent_forge/consciousness/perturb/harness.py`
- Harness now registers active perturbations on kernel for module consumption.

7. Workspace summary extensions
- `agent_forge/src/agent_forge/core/workspace.py`
- Added:
  - source contribution table
  - source gini
  - ignition burst metrics

8. Continuity regression tests
- `agent_forge/tests/test_consciousness_continuity.py`
- Coverage:
  - module tick cadence
  - persistent state across kernel restart
  - working set continuity output
  - reaction trace + ignition
  - attention drop perturbation behavior

## Remaining Work (Next Tranche)

1. World model predictive coding v1
- Move from event-type transition model to feature-space prediction with latent belief state.

2. Memory bridge integration
- Add memory-forge introspection/recall ingestion into attention candidate generation.

3. Ablation benchmark suite
- Add module-disable trial matrix and expected-delta assertions for ignition, RCI, agency, and report groundedness.

4. Simulation mode runtime
- Add explicit simulated percept stream with mode-aware report labeling.

## Validation Commands

```sh
PYTHONPATH=lib:agent_forge/src:eidos_mcp/src:crawl_forge/src \
./eidosian_venv/bin/python -m pytest -q \
  agent_forge/tests/test_consciousness_continuity.py \
  agent_forge/tests/test_consciousness_milestone_a.py \
  agent_forge/tests/test_consciousness_milestone_b.py \
  agent_forge/tests/test_consciousness_milestone_d.py \
  agent_forge/tests/test_consciousness_trials.py \
  agent_forge/tests/test_consciousness_benchmarks.py \
  agent_forge/tests/test_consciousness_integrated_benchmark.py
```
