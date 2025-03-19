# 🔄 Version Forge 🛠️

> _"Version information isn't bookkeeping—it's structural integrity across time."_

A type-safe toolkit for semantic versioning with cross-component compatibility enforcement. Version Forge treats version numbers as what they truly are: mathematical contracts encoding compatibility guarantees through time and space.

```ascii
  ╭───────────────────────────────────╮
  │  RECURSIVE VERSIONING HARMONY     │
  │  ↺        ↻        ↺        ↻     │
  │    ↻    versions    ↺             │
  ╰───────────────────────────────────╯
```

[![Version](https://img.shields.io/badge/Version-3.14.7-blue)](https://github.com/Ace1928/version_forge) [![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT) [![Python](https://img.shields.io/badge/Python-3.12+-purple)](https://www.python.org/)

## ⚡ Core Capabilities

- **Semantic Version Control** — Parse, validate, and manipulate versions with type safety
- **Cross-Component Compatibility** — Ensure version coherence across system boundaries
- **Migration Intelligence** — Generate migration guides with effort estimation
- **Automated Version Management** — Update version strings across file types
- **CLI Interface** — Command-line tools for version operations

```ascii
╭──────────────────────────────────────────────────────╮
│ (￣ω￣) "Versions should tell the truth, even when   │
│         the rest of your documentation is lying."    │
╰──────────────────────────────────────────────────────╯
```

## 🔧 Installation

```bash
# Install from source
git clone https://github.com/Ace1928/version_forge.git
cd version_forge
pip install -e .

# Or via pip when published
pip install version-forge
```

## 🚀 Usage Examples

### Basic Version Handling

```python
from version_forge import parse_version, format_version
from typing import Optional

# Parse a version string into its structured components
version = parse_version("1.2.3-beta.4")  # Type: VersionInfo
print(version)  # VersionInfo(major=1, minor=2, patch=3, label="beta", label_num=4)

# Format a version object back to string with perfect fidelity
formatted: str = format_version(version)
print(formatted)  # "1.2.3-beta.4"
```

### Version Compatibility Checking

```python
from version_forge import is_compatible, calculate_delta
from typing import Dict, Any

# Check if versions are compatible (returns bool)
compatible: bool = is_compatible("2.0.0", "2.1.5")  # True (minor version change)
breaking: bool = is_compatible("2.0.0", "3.0.0")    # False (major version change)

# Calculate the semantic delta between versions
delta: Dict[str, Any] = calculate_delta("1.0.0", "2.0.0")
print(delta)  # {"type": "major", "breaking": True, "distance": 1}
```

### Migration Guide Generation

```python
from version_forge import MigrationGuideGenerator
from typing import Dict, Any, List

# Create migration paths between versions
generator = MigrationGuideGenerator()
guide: Dict[str, Any] = generator.generate_migration_guide("mylib", "1.0.0", "2.0.0")

# Extract the information that matters
print(f"Upgrade type: {guide['upgrade_type']}")         # "major"
print(f"Effort level: {guide['estimated_effort']}")     # "significant"
print(f"Breaking changes: {guide['breaking_changes']}") # List[BreakingChange]
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

### Version Semantics

```ascii
┌─────────────────────────────────────────────────────────────┐
│ MAJOR.MINOR.PATCH-LABEL.NUM+BUILD                           │
│                                                             │
│ MAJOR - Breaking changes that demand migration rituals      │
│ MINOR - New features that preserve existing contracts       │
│ PATCH - Corrections to unintended behaviors                 │
│ LABEL - Warning labels for the chronically adventurous      │
└─────────────────────────────────────────────────────────────┘
```

## 📚 Resources

- [Semantic Versioning Specification](https://semver.org/) - The mathematical foundation
- [Documentation](https://github.com/Ace1928/version_forge/docs) - Full usage guide
- [Issue Tracker](https://github.com/Ace1928/version_forge/issues) - Report version anomalies

---

Maintained with recursive precision by [Lloyd Handyside](mailto:ace1928@gmail.com) and [Neuroforge](https://neuroforge.io)
© 3.14.7 - The irrational version for rational minds

> "The only thing more dangerous than a missing version number is one that lies about compatibility." — Eidos
