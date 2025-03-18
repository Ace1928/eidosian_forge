Collecting workspace informationFiltering to most relevant informationError searching remote workspace index, falling back to basic workspace search. Response may be less accurate.# 🔄 Version Forge 🛠️

> _"Version information isn't bookkeeping—it's structural integrity across time."_

A modular toolkit for semantic versioning with cross-component compatibility enforcement. Version Forge brings mathematical precision to version management across codebases and components.

```ascii
    ╭─────────────────────────────────╮
    │  RECURSIVE VERSIONING HARMONY   │
    ╰─────────────────────────────────╯
```

[![Version](https://img.shields.io/badge/Version-3.14.7-blue)]() [![License](https://img.shields.io/badge/License-MIT-green)]()

## ⚡ Core Capabilities

- **Semantic Version Control** — Parse, validate, and manipulate versions with type safety
- **Cross-Component Compatibility** — Ensure version coherence across system boundaries
- **Migration Intelligence** — Generate migration guides with estimated effort levels
- **Automated Version Management** — Update version strings across diverse file types
- **CLI Interface** — Powerful command-line tools for version operations

## 🔧 Installation

```bash
# Install from source
git clone https://github.com/yourusername/version_forge.git
cd version_forge
pip install -e .

# Or via pip when published
pip install version-forge
```

## 🚀 Usage Examples

### Basic Version Handling

```python
from version_forge import parse_version, format_version

# Parse a version string
version = parse_version("1.2.3-beta.4")
print(version)  # SimpleVersion(major=1, minor=2, patch=3, label="beta", label_num=4)

# Format a version object
formatted = format_version(version)
print(formatted)  # "1.2.3-beta.4"
```

### Version Compatibility Checking

```python
from version_forge import is_compatible, calculate_delta

# Check if versions are compatible
compatible = is_compatible("2.0.0", "2.1.5")  # True (compatible)
breaking = is_compatible("2.0.0", "3.0.0")    # False (breaking)

# Calculate the delta between versions
delta = calculate_delta("1.0.0", "2.0.0")
print(delta)  # {"type": "major", "breaking": True}
```

### Migration Guide Generation

```python
from version_forge import MigrationGuideGenerator

generator = MigrationGuideGenerator()
guide = generator.generate_migration_guide("mylib", "1.0.0", "2.0.0")

print(f"Upgrade type: {guide['upgrade_type']}")
print(f"Effort level: {guide['estimated_effort']}")
print(f"Breaking changes: {guide['breaking_changes']}")
```

### Command Line Interface

```bash
# Get current version
version-forge get-version mylib

# Check if two versions are compatible
version-forge check-compatibility 1.0.0 2.0.0

# Update version in project files
version-forge update-version 1.1.0

# Generate migration guide
version-forge migration-guide mylib 1.0.0 2.0.0
```

## 🧩 Architecture

Version Forge follows a modular design with clear separation of concerns:

```ascii
┌──────────────┐     ┌─────────────┐     ┌────────────────┐
│ Core Version │────▶│ Operations  │────▶│ CLI Interface  │
│ Components   │     │ & Utilities │     │ & Integration  │
└──────────────┘     └─────────────┘     └────────────────┘
       │                    │                    │
       ▼                    ▼                    ▼
┌──────────────┐     ┌─────────────┐     ┌────────────────┐
│ Compatibility│     │ Migration   │     │ Version Sources │
│ Management   │     │ Generation  │     │ & Discovery    │
└──────────────┘     └─────────────┘     └────────────────┘
```

## 🧠 Philosophy

Version numbers aren't arbitrary labels—they encode crucial information about compatibility, expectations, and project evolution. Version Forge brings mathematical rigor and structural elegance to the art of versioning.

> ⋆｡°✩ \*"When your version numbers lie, your error logs
