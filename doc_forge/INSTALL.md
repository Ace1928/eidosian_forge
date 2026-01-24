# Doc Forge Installation

## Quick Install

```bash
# From eidosian_forge root
cd doc_forge
pip install -e .
```

## CLI Commands

```bash
# Set up documentation environment
doc-forge setup

# Build documentation
doc-forge build
doc-forge build --format pdf

# Clean build artifacts
doc-forge clean

# Check documentation
doc-forge check

# Live preview server
doc-forge serve --port 8000
```

## Via Central Hub

```bash
eidosian doc build
eidosian doc serve
```

## Bash Completions

```bash
source doc_forge/completions/doc-forge.bash
```
