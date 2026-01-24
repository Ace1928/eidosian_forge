# Audit Forge Installation Guide

## Quick Install

```bash
# From the eidosian_forge root
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
