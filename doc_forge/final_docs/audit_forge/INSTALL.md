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

# Audit Forge Installation Guide

## Quick Install

```bash
pip install -e ./audit_forge
```

## Dependencies

- Python >=3.12
- typer
- rich

## Verify Installation

```bash
audit-forge --help
```

## CLI Usage

### Check Audit Coverage

```bash
# Check coverage for current directory
audit-forge coverage

# Check coverage for specific path
audit-forge coverage /path/to/project
```

### Mark Files as Reviewed

```bash
# Mark a file as reviewed by user
audit-forge mark path/to/file.py

# Mark with specific agent
audit-forge mark path/to/file.py --agent eidos
```

### Add TODO Items

```bash
# Add a task to immediate section
audit-forge todo "Fix the bug"

# Add to specific section
audit-forge todo "Refactor module" --section "Short-Term"
```

## Bash Completion

Add to your `~/.bashrc`:
```bash
source /path/to/eidosian_forge/audit_forge/completions/audit-forge.bash
```

## Python API

```python
from audit_forge import AuditForge

# Initialize
audit = AuditForge(data_dir="./audit_data")

# Check coverage
stats = audit.verify_coverage("./src")
print(f"Unreviewed: {stats['unreviewed_count']}")

# Mark as reviewed
audit.coverage.mark_reviewed("file.py", agent_id="user")
```

## Integration with Eidosian Forge

When installed as part of the Eidosian Forge ecosystem:

```bash
# Via central hub
eidosian audit --help

# Direct CLI
audit-forge --help
```

## Data Storage

Audit data is stored in `audit_data/` directory:
- `coverage.json` - File review status
- `todo.json` - TODO items
- `audit_log.json` - Audit history

---

*Part of the Eidosian Forge - Code audit and coverage tracking.*



[ Prompt: 52.8 t/s | Generation: 7.4 t/s ]

Exiting...