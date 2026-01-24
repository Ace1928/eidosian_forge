# 📦 Version Forge Installation

Semantic versioning utilities with bump, comparison, and validation.

## Quick Install

```bash
# Clone (if standalone)
git clone https://github.com/Ace1928/version_forge.git
cd version_forge

# Install
pip install -e .

# Or as part of Eidosian Forge
pip install -e ./version_forge
```

## Verify Installation

```bash
python -c "from version_forge import Version; print(Version('1.2.3'))"
```

## CLI Usage

```bash
# Bump version
version-forge bump 1.2.3 minor  # -> 1.3.0
version-forge bump 1.2.3 patch  # -> 1.2.4
version-forge bump 1.2.3 major  # -> 2.0.0

# Compare versions
version-forge compare 1.2.3 1.2.4  # 1.2.4 is newer

# Validate
version-forge validate 1.2.3  # Valid semver

# Help
version-forge --help
```

## Python API

```python
from version_forge import Version, bump_version

v = Version("1.2.3")
print(v.major, v.minor, v.patch)  # 1 2 3

new_v = bump_version(v, "minor")
print(new_v)  # 1.3.0
```

## Dependencies

- No external dependencies (stdlib only)
- `eidosian_core` - Universal decorators and logging

