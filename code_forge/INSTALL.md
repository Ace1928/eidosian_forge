# Code Forge Installation Guide

## Quick Install

```bash
# From the code_forge directory
pip install -e .

# Or install with all extras
pip install -e ".[dev,refactor,lint]"
```

## Verify Installation

```bash
# Check CLI is available
code-forge --version

# Check system status
code-forge status
```

## Enable Bash Completions

```bash
# Add to your ~/.bashrc
source /home/lloyd/eidosian_forge/code_forge/completions/code-forge.bash

# Or for a one-time session
source completions/code-forge.bash
```

## Standalone vs. Integrated Usage

### Standalone
Code Forge works independently with these capabilities:
- Python file AST analysis
- Code element indexing (functions, classes, methods)
- Code library for storing snippets
- Code search by name or docstring

### Enhanced with Other Forges
When other forges are installed, additional capabilities are enabled:

| Forge | Enhanced Capability |
|-------|---------------------|
| `knowledge_forge` | Semantic code understanding |
| `llm_forge` | AI-powered code analysis |

Check available integrations:
```bash
code-forge integrations
```

## Configuration

Data is stored at:
```
/home/lloyd/eidosian_forge/data/
├── code_index.json    # Indexed code elements
└── code_library.json  # Code snippets
```

## CLI Commands

```bash
code-forge status              # Show index status
code-forge analyze <file>      # Analyze Python file
code-forge index <path>        # Index codebase
code-forge search <query>      # Search code elements
code-forge ingest <file>       # Add to code library
code-forge library             # List library contents
code-forge stats               # Detailed statistics
```

## Python API

```python
from code_forge import CodeAnalyzer, CodeIndexer, CodeLibrarian

# Analyze a file
analyzer = CodeAnalyzer()
result = analyzer.analyze_file("example.py")
print(f"Functions: {len(result['functions'])}")

# Index codebase
indexer = CodeIndexer()
count = indexer.index_directory("/path/to/code")
print(f"Indexed {count} elements")

# Search code
for elem in indexer.elements.values():
    if "memory" in elem.name.lower():
        print(f"{elem.element_type}: {elem.qualified_name}")

# Code library
librarian = CodeLibrarian()
sid = librarian.add_snippet(code_content, metadata=analysis)
```

## Troubleshooting

### Import Errors
Ensure the lib directory is in your Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:/home/lloyd/eidosian_forge/lib"
```

### Slow Indexing
Large codebases may take time to index. Use `--json` for progress:
```bash
code-forge index /path --json
```
