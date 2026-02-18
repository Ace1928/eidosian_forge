Loading model... 


▄▄ ▄▄
██ ██
██ ██  ▀▀█▄ ███▄███▄  ▀▀█▄    ▄████ ████▄ ████▄
██ ██ ▄█▀██ ██ ██ ██ ▄█▀██    ██    ██ ██ ██ ██
██ ██ ▀█▄██ ██ ██ ██ ▀█▄██ ██ ▀████ ████▀ ████▀
                                    ██    ██
                                    ▀▀    ▀▀

build      : b8083-ae2d3f28a
model      : qwen2.5-1.5b-instruct-q5_k_m.gguf
modalities : text

available commands:
  /exit or Ctrl+C     stop or exit
  /regen              regenerate the last response
  /clear              clear the chat history
  /read               add a text file


> <|im_start|>system
You are an Eidosian Documentation Forge agent. Your task is to generate detailed, accurate, and structurally elegant documentation for the provided source file.
Follow these requirements:
1. File Summary: High-level overview of what the file does.
2. API Documentation: List all functions, classes, and their parameters/methods.
3. Current Status: Accurate assessment of the file's current state based on its content and comments.
4. Potential Future Directions: Eidosian ideas for ... (truncated)

# File Summary

The `policies.yaml` file contains configuration for the autonomy ladder and risk model, as well as confidence parameters.

# API Documentation

### Autonomy Ladder Policies

| Policy Name | Action | Require Approvals | Minimum Competence | Maximum Risk |
|-------------|--------|------------------|--------------------|--------------|
| observe     | False  | None             | 0.0                | 1.0         |
| propose     | False  | None             | 0.3                | 0.8         |
| act_limited  | True   | Commands          | 0.5                | 0.6         |
| act_full    | True   | None             | 0.7                | 0.4         |

### Risk Model

| Parameters       | Weighting (Proportion) | Cutoffs (Thresholds) |
|------------------|------------------------|-----------------------|
| Surface Area     | 35%                   | Red: 0.7, Amber: 0.4 |
| Reversibility      | 25%                   |                        |
| Blast Radius       | 25%                   |                        |
| Novelty            | 15%                   |                        |

### Confidence Parameters

| Parameter          | Value (Units)    | Decay per Week |
|--------------------|-------------------|----------------|
| Minimum Brier Improvement | 0.02             | 0.01           |

### Approvals

#### Safe Commands
- `pnpm test -w`
- `pnpm lint --fix`
- `ruff --fix`
- `pytest -q`
- `go test ./...`

#### Path Allowlist
- `/src`
- `/tests`
- `/docs`
- `/tools`

#### Deny Patterns
- `/etc`
- `/var`
- `~

[ Prompt: 49.5 t/s | Generation: 7.0 t/s ]

Exiting...