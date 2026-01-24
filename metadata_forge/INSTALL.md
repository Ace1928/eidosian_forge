# Metadata Forge Installation Guide

## Quick Install

```bash
# From the eidosian_forge root
pip install -e ./metadata_forge
```

## Dependencies

- Python >=3.12

## Verify Installation

```bash
metadata-forge version
```

## CLI Usage

### Generate Metadata Template

```bash
# Generate template to stdout
metadata-forge template

# Generate to file
metadata-forge template -o metadata.json
```

### Validate Metadata

```bash
# Validate a metadata file
metadata-forge validate metadata.json

# Validate YAML
metadata-forge validate config.yaml
```

### Version Info

```bash
metadata-forge version
```

## Metadata Schema

The Eidosian metadata schema includes:

```json
{
  "name": "project-name",
  "version": "1.0.0",
  "description": "Project description",
  "authors": [
    {"name": "Author", "email": "author@example.com"}
  ],
  "eidosian": {
    "principles": ["precision", "elegance", "flow"],
    "forge": "metadata_forge"
  }
}
```

## Bash Completion

Add to your `~/.bashrc`:
```bash
source /path/to/eidosian_forge/metadata_forge/completions/metadata-forge.bash
```

## Python API

```python
from metadata_forge.schema import (
    create_metadata_template,
    validate_metadata
)

# Create template
template = create_metadata_template()

# Validate
errors = validate_metadata(my_metadata)
if errors:
    print("Validation errors:", errors)
```

## Integration with Eidosian Forge

When installed as part of the Eidosian Forge ecosystem:

```bash
# Via central hub
eidosian metadata --help

# Direct CLI
metadata-forge --help
```

---

*Part of the Eidosian Forge - Eidosian metadata management.*
