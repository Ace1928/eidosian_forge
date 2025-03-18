# 🔄 Version Forge 🛠️

> _"Version information isn't bookkeeping—it's structural integrity across time."_

A type-safe toolkit for semantic versioning with cross-component compatibility enforcement. Version Forge treats version numbers as what they truly are: mathematical contracts that encode compatibility guarantees.

```ascii
  ╭───────────────────────────────────╮
  │  RECURSIVE VERSIONING HARMONY     │
  │  ↺        ↻        ↺        ↻     │
  │    ↻    versions    ↺             │
  ╰───────────────────────────────────╯
```

[![Version](https://img.shields.io/badge/Version-3.14.7-blue)](https://github.com/yourusername/version_forge) [![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)

## ⚡ Core Capabilities

- **Semantic Version Control** — Parse, validate, and manipulate versions with type safety
- **Cross-Component Compatibility** — Ensure version coherence across system boundaries
- **Migration Intelligence** — Generate migration guides with effort estimation
- **Automated Version Management** — Update version strings across file types
- **CLI Interface** — Command-line tools for version operations

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

# Parse a version string into its structured components
version = parse_version("1.2.3-beta.4")
print(version)  # SimpleVersion(major=1, minor=2, patch=3, label="beta", label_num=4)

# Format a version object back to string
formatted = format_version(version)
print(formatted)  # "1.2.3-beta.4"
```

### Version Compatibility Checking

```python
from version_forge import is_compatible, calculate_delta

# Check if versions are compatible (returns bool)
compatible = is_compatible("2.0.0", "2.1.5")  # True (minor version change)
breaking = is_compatible("2.0.0", "3.0.0")    # False (major version change)

# Calculate the semantic delta between versions
delta = calculate_delta("1.0.0", "2.0.0")
print(delta)  # {"type": "major", "breaking": True, "distance": 1}
```

### Migration Guide Generation

```python
from version_forge import MigrationGuideGenerator

# Create migration paths between versions
generator = MigrationGuideGenerator()
guide = generator.generate_migration_guide("mylib", "1.0.0", "2.0.0")

print(f"Upgrade type: {guide['upgrade_type']}")         # "major"
print(f"Effort level: {guide['estimated_effort']}")     # "significant"
print(f"Breaking changes: {guide['breaking_changes']}") # [...]
```

### Command Line Interface

```bash
# Get current version (when you can't remember what version of reality you're on)
version-forge get-version mylib

# Check if two versions play nicely together
version-forge check-compatibility 1.0.0 2.0.0

# Update version in project files (let the computer do the tedious work)
version-forge update-version 1.1.0

# Generate migration guide (aka "what will break and how badly?")
version-forge migration-guide mylib 1.0.0 2.0.0
```

## 🧩 Architecture

Version Forge follows a modular design with fractal precision:

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

Version numbers aren't arbitrary labels—they encode crucial information about compatibility, expectations, and project evolution. When versions lie, systems die.

```ascii
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ "When your version numbers lie, your error logs     ┃
┃  become mystery novels nobody wants to read."       ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

## 📚 Resources

- [Semantic Versioning Specification](https://semver.org/) - The mathematical foundation
- [Documentation](https://github.com/yourusername/version_forge/docs) - Full usage guide
- [Issue Tracker](https://github.com/yourusername/version_forge/issues) - Report version anomalies

---

_Made with recursive precision_ ⋆｡°✩
