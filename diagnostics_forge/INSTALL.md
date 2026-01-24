# 🔬 Diagnostics Forge Installation

System diagnostics, health checks, and monitoring utilities.

## Quick Install

```bash
pip install -e ./diagnostics_forge

# Verify
python -c "from diagnostics_forge import SystemDiagnostics; print('✓ Ready')"
```

## CLI Usage

```bash
# Run system diagnostics
diagnostics-forge check

# Get system info
diagnostics-forge info

# Monitor resources
diagnostics-forge monitor --interval 5

# Health check
diagnostics-forge health

# Help
diagnostics-forge --help
```

## Python API

```python
from diagnostics_forge import SystemDiagnostics, run_health_check

# Run diagnostics
diag = SystemDiagnostics()
report = diag.full_check()
print(report)

# Quick health check
health = run_health_check()
print(f"Status: {health.status}")
```

## Dependencies

- `psutil` - System monitoring
- `eidosian_core` - Universal decorators and logging

