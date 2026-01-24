# 🔧 Repo Forge Installation

Repository analysis, management, and automation tools.

## Quick Install

```bash
pip install -e ./repo_forge

# Verify
python -c "from repo_forge import RepoAnalyzer; print('✓ Ready')"
```

## CLI Usage

```bash
# Analyze repository
repo-forge analyze .

# Generate structure report
repo-forge structure --output report.md

# Check dependencies
repo-forge deps

# Help
repo-forge --help
```

## Python API

```python
from repo_forge import RepoAnalyzer

# Analyze current directory
analyzer = RepoAnalyzer(".")
report = analyzer.full_analysis()
print(report)
```

## Dependencies

- `gitpython` - Git repository access
- `eidosian_core` - Universal decorators and logging

