# Part 27: Self-Upgrading Loop Architecture

## Goal

Add a safe, falsifiable self-upgrade loop where the runtime proposes parameter changes, runs controlled trials, and commits only improvements that pass guardrails.

## Runtime Loop

1. Read current tuned overlay from persistent state.
2. Propose a candidate overlay mutation.
3. Run micro-trials with deterministic seed and captured artifacts.
4. Compute objective score and guardrail checks.
5. Commit tuned overlay if score improves and safety checks pass; otherwise rollback.

## Core Components

- `agent_forge/src/agent_forge/consciousness/tuning/params.py`
- `agent_forge/src/agent_forge/consciousness/tuning/overlay.py`
- `agent_forge/src/agent_forge/consciousness/tuning/optimizer.py`
- `agent_forge/src/agent_forge/consciousness/modules/autotune.py`
- `agent_forge/src/agent_forge/consciousness/bench/trials.py`
- `agent_forge/src/agent_forge/consciousness/kernel.py`

## Event Contract

- `tune.baseline`: baseline score initialization.
- `tune.proposed`: candidate mutation proposal.
- `tune.commit`: accepted tuned overlay.
- `tune.rollback`: rejected candidate and reason.
- `tune.skipped`: unsafe runtime state or cooling window.

## Guardrail Contract

Trial report now exposes:

- `module_error_count`
- `degraded_mode_ratio`
- `winner_count`
- `ignitions_without_trace`

Commit gate requires:

- module errors <= configured threshold
- degraded ratio <= configured threshold
- winner count <= configured threshold
- ignition trace violations <= configured threshold
- objective improvement >= `autotune_min_improvement`

## Rollback Properties

- Overlay commit is versioned (`tuned_overlay_version`).
- Full overlay history is preserved (`tuned_overlay_history`).
- Kernel resolves config in strict precedence:
  `runtime_overrides > tuned_overlay > base_config > defaults`.

## Acceptance Criteria

1. Autotune can run unattended without destabilizing the daemon loop.
2. Every committed change is traceable to trial ID + score + overlay version.
3. Invalid overlay keys are rejected and logged (`consciousness.param_invalid`).
4. Kernel can apply runtime trial overrides without polluting persistent overlays.

