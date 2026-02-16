# Part 17: Consciousness Trial Runner (CTR)

Status: core CTR package delivered in Stage H PR-H3 (`trials.py`, `tasks.py`, `scoring.py`, `reporting.py` + tests).

## Goal

Transform ad hoc testing into reproducible, persisted, benchmark-grade experiments.

## Deliverables

1. standardized `TrialSpec`
2. deterministic trial execution lifecycle
3. persisted artifacts and summary reports
4. task-aware trial support

## New Package

- `agent_forge/src/agent_forge/consciousness/bench/`

Files:

- `trials.py`
- `tasks.py`
- `scoring.py`
- `reporting.py`
- `ablations.py` (pending PR-H5)
- `golden.py` (pending PR-H5)

## TrialSpec Contract

```python
@dataclass
class TrialSpec:
    name: str
    warmup_beats: int
    baseline_s: float
    perturb_s: float
    recovery_s: float
    task: str | None
    perturbations: list[dict[str, Any]]
    disable_modules: list[str]
    seed: int
```

## Execution Lifecycle

1. initialize kernel with seed and config overrides
2. warm-up beats
3. collect baseline window
4. apply perturbation set
5. collect perturb window
6. collect recovery window
7. compute metrics and task score
8. persist all artifacts
9. emit `bench.trial_result`

## Artifact Layout

`<state_dir>/consciousness/bench/<ts>_<trial>_<hash>/`

- `spec.json`
- `metrics.jsonl`
- `events_window.jsonl`
- `report.json`
- `summary.md`

## Scoring Domains

- dynamics: ignition trace, coherence, RCI
- self-binding: agency/boundary stability
- grounding: report evidence consistency
- task success: objective task validators

## Tasks API

`tasks.py` defines:

- simple deterministic tasks (classification, controllability discrimination)
- module-integrated tasks (winner consistency, report grounding)
- environment-backed tasks (future mock world)

## Validation

### Determinism

- same spec + seed yields bounded metric variance

### Replay

- replaying saved events reproduces score outputs

### Robustness

- malformed spec rejected with explicit schema errors

## Acceptance Criteria

1. one-command trial execution from CLI and MCP.
2. persisted trial artifacts are self-describing and replayable.
3. CI can run reduced trial matrix and enforce non-regression gates.
