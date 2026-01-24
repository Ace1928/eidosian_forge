# Refactor Forge Installation Guide

## Quick Install

```bash
# From the eidosian_forge root
pip install -e ./refactor_forge
```

## Dependencies

- Python >=3.12

## Verify Installation

```bash
refactor-forge --version
```

## CLI Usage

### Analyze Code

```bash
# Analyze a Python file
refactor-forge script.py --analyze-only

# Verbose analysis
refactor-forge script.py --analyze-only -v
```

### Refactor Code

```bash
# Refactor to new package
refactor-forge script.py -o ./output_dir

# With custom package name
refactor-forge script.py -n my_package

# Dry run (show what would happen)
refactor-forge script.py --dry-run
```

### Options

| Option | Description |
|--------|-------------|
| `-o, --output-dir` | Output directory for refactored package |
| `-n, --package-name` | Custom package name |
| `--analyze-only` | Only analyze, don't refactor |
| `--dry-run` | Show what would be done |
| `-v, --verbose` | Verbose output |
| `--clean` | Clean output directory first |

## Bash Completion

Add to your `~/.bashrc`:
```bash
source /path/to/eidosian_forge/refactor_forge/completions/refactor-forge.bash
```

## Python API

```python
from refactor_forge.analyzer import analyze_code
from refactor_forge.reporter import print_analysis_report

# Analyze
analysis = analyze_code("script.py")
print_analysis_report(analysis)
```

## Integration with Eidosian Forge

When installed as part of the Eidosian Forge ecosystem:

```bash
# Via central hub
eidosian refactor --help

# Direct CLI
refactor-forge --help
```

---

*Part of the Eidosian Forge - Transform code into perfect modular architecture.*
