# 📁 File Forge Installation

File operations, directory trees, content search, and duplicate detection.

## Quick Install

```bash
pip install -e ./file_forge

# Verify
python -c "from file_forge import find_files, find_duplicates; print('✓ Ready')"
```

## CLI Usage

```bash
# Find files by pattern
file-forge find "*.py" --path ./src

# Search content
file-forge search "TODO" --path ./src --glob "*.py"

# Find duplicates
file-forge duplicates ./data

# Print directory tree
file-forge tree ./project

# Sync directories
file-forge sync ./source ./backup

# Help
file-forge --help
```

## Python API

```python
from file_forge import find_files, find_duplicates, search_content

# Find all Python files
py_files = find_files("*.py", path="./src")

# Search for pattern in files
matches = search_content("def main", glob="*.py")

# Find duplicate files
dups = find_duplicates("./data")
```

## Dependencies

- `eidosian_core` - Universal decorators and logging

