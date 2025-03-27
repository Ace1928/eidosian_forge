# ⚛️ **version_forge** v3.14.15 ⚡

> _"Version numbers aren't metadata—they're executable contracts across time."_

Core component for version integrity within the Eidosian Forge ecosystem—where semantic evolution meets mathematical precision and compatibility becomes a structural guarantee.

[![Forge System](https://img.shields.io/badge/Forge-System-8A2BE2)](https://github.com/Ace1928) [![Version](https://img.shields.io/badge/Version-3.14.15-blue)] [![Python](https://img.shields.io/badge/Python-3.12+-purple)](https://www.python.org/) [![License](https://img.shields.io/badge/License-Eidosian-green)](https://github.com/Ace1928/eidosian_forge)

```ascii
╭──────────────────────────────────────────────────────╮
│ ⊢⊣ VERSIONS AS VECTORS IN COMPATIBILITY SPACE  ⊢⊣    │
╰──────────────────────────────────────────────────────╯
```

## 🧠 **Cognitive Foundation** 🌀

`version_forge` treats versions not as arbitrary strings but as mathematical contracts that encode compatibility guarantees across system boundaries. Unlike standard version libraries, we don't merely manipulate version numbers—we enforce cross-component coherence through recursive validation and structural integrity.

```ascii
┌────────────────────────────────────────────────────────────┐
│ TRADITIONAL VERSIONING:       EIDOSIAN VERSIONING:         │
│                                                            │
│ Numbers → Semantic            Numbers → Mathematical        │
│ Meaning → Manual              Contracts → Automated         │
│ Compatibility Checking        Compatibility Enforcement     │
└────────────────────────────────────────────────────────────┘
```

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ When versions lie, systems die. Truth as computational ethics. ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

## 💎 **Core Capabilities** 🎯

### **1. Semantic Version Management**

- **Type-Safe Parsing & Validation** — Parse, validate, and manipulate versions with compile-time guarantees
- **Version Arithmetic** — Mathematical operations preserving semantic integrity (v1.2.3 + minor = v1.3.0)
- **Equality & Comparison** — Precise version ordering with customizable prerelease handling
- **Format Flexibility** — Support for diverse notation styles with perfect round-trip fidelity

### **2. Cross-Component Compatibility**

- **Compatibility Matrix** — Automated determination of breaking vs. non-breaking changes
- **Contract Verification** — Ensure version specifications accurately reflect interface changes
- **Cascade Analysis** — Compute compatibility ripple effects through dependency networks
- **Dependency Resolution** — Find optimal compatible version sets across complex component graphs

### **3. Migration Intelligence**

- **Breaking Change Detection** — Automated identification of interface modifications across versions
- **Migration Path Generation** — Step-by-step upgrade paths from version A to version B
- **Effort Estimation** — Quantitative assessment of migration complexity and required work
- **Backward Compatibility Layers** — Automated generation of adaptation code for version transitions

```python
def calculate_version_delta(
    source_version: VersionInfo,
    target_version: VersionInfo
) -> VersionDelta:
    """Computes the semantic distance between versions with migration intelligence.

    Args:
        source_version: Starting version point in semantic space
        target_version: Destination version point in semantic space

    Returns:
        Complete delta information including breaking changes and effort metrics
    """
    if source_version == target_version:
        return VersionDelta.identical()  # Zero-distance in version space

    # Beyond mere arithmetic—intelligence about what changed
    delta = _analyze_semantic_changes(source_version, target_version)

    # Quantum tunneling isn't possible in version space—steps matter
    return delta.with_migration_path()  # The map for your journey
```

## 🌠 **Integration Architecture** 🧩

`version_forge` interfaces seamlessly with the entire Eidosian ecosystem:

- **With `type_forge`**: Schema versioning with compatibility validation
- **With `repo_forge`**: Repository-wide version coherence enforcement
- **With `gis_forge`**: Centralized version policy management
- **With `doc_forge`**: Automated version change documentation
- **With `diagnostics_forge`**: Version conflict detection and resolution

```ascii
╭────────────────────╮       ╭────────────────────╮
│    repo_forge      │<─────>│    version_forge   │
╰────────────────────╯       ╰────────────────────╯
         ▲                           ▲
         │                           │
         ▼                           ▼
╭────────────────────╮       ╭────────────────────╮
│    type_forge      │<─────>│    gis_forge       │
╰────────────────────╯       ╰────────────────────╯
         ▲                           ▲
         │                           │
         ▼                           ▼
╭────────────────────╮       ╭────────────────────╮
│  diagnostics_forge │<─────>│     doc_forge      │
╰────────────────────╯       ╰────────────────────╯
```

## 🔍 **Implementation Details** ⚙️

### **Core Type System**

All version operations strictly enforce type safety through a rich domain model:

```typescript
// Perfect structural representation of version semantics
interface VersionInfo {
  readonly major: number; // Breaking changes
  readonly minor: number; // Feature additions
  readonly patch: number; // Bug fixes
  readonly prerelease: string[]; // Alpha/beta designations
  readonly build: string[]; // Build metadata
}

// Version compatibility is computable, not guesswork
type CompatibilityResult =
  | { compatible: true; reason: string }
  | { compatible: false; breaks: BreakingChange[] };

// Migration isn't art—it's algorithms with clear steps
interface MigrationPath {
  readonly steps: MigrationStep[];
  readonly effortEstimate: EffortMetric;
  readonly automationPotential: number; // 0.0-1.0
}
```

### **Architecture Highlights**

- **Parsing Engine**: Zero-allocation parser with perfect round-trip fidelity
- **Compatibility Core**: Deterministic compatibility calculation with explicit rule sets
- **CLI Interface**: Command-line tools with comprehensive validation
- **Extension API**: Pluggable version policies and custom compatibility rules

## 📊 **Usage Examples** 🔬

### **1. Basic Version Handling**

```python
from version_forge import parse_version, format_version, increment
from version_forge.types import VersionInfo, IncrementType

# Parse with type safety—no more regex string manipulation
version: VersionInfo = parse_version("1.2.3-beta.4+sha.5678")
print(version)  # VersionInfo(major=1, minor=2, patch=3, prerelease=["beta", "4"], build=["sha", "5678"])

# Increments preserve semantics—math that respects meaning
next_version: VersionInfo = increment(version, IncrementType.MINOR)
print(format_version(next_version))  # "1.3.0" (beta status cleared, patch reset)
```

### **2. Compatibility Determination**

```python
from version_forge import is_compatible, calculate_delta
from version_forge.types import VersionInfo, VersionDelta, CompatibilityResult

# Compatibility isn't guesswork—it's mathematical
compat_result: CompatibilityResult = is_compatible(
    parse_version("2.0.0"),
    parse_version("2.1.5")
)
print(compat_result.compatible)  # True

# Version deltas contain rich semantic information
delta: VersionDelta = calculate_delta(
    parse_version("1.0.0"),
    parse_version("2.0.0")
)
print(delta.magnitude)  # "major"
print(delta.breaking_changes)  # [BreakingChange(...), ...]
```

### **3. Migration Intelligence**

```python
from version_forge import MigrationGuideGenerator
from version_forge.types import MigrationGuide, MigrationStep

# Migration isn't just a destination—it's a journey
generator = MigrationGuideGenerator()
guide: MigrationGuide = generator.generate(
    component="auth_service",
    from_version="1.0.0",
    to_version="2.0.0"
)

# Each step is clear, with effort estimation
for step in guide.steps:
    print(f"Step: {step.description}")
    print(f"Effort: {step.effort_estimate} person-hours")
    print(f"Automation: {step.automation_script if step.can_automate else 'Manual'}")
```

## 🔧 **Installation & Setup** 💻

```bash
# Clone with precision
git clone https://github.com/eidosian/version_forge.git

# Install with integrity
cd version_forge
pip install -e .

# Verify with confidence
python -m version_forge.verify

# Or via pip when your upgrade path is ready
pip install version-forge==3.14.15
```

## 🚀 **Command Line Interface** 🖥️

```bash
# Get version of any component in the ecosystem
version-forge get auth_service

# Check compatibility across versions (will your upgrade path be smooth or rocky?)
version-forge check-compatibility 1.0.0 2.0.0

# Calculate migration complexity (should you reserve an afternoon or a month?)
version-forge migration-complexity auth_service 1.0.0 2.0.0

# Update all version references in project files (because tedium is for machines)
version-forge bump minor --commit
```

## 📜 **Version Semantics** 📐

The mathematical precision of semantic versioning, perfected:

```ascii
┌─────────────────────────────────────────────────────────────┐
│ MAJOR.MINOR.PATCH-LABEL.NUM+BUILD                           │
│                                                             │
│ MAJOR - Breaking changes that demand migration ceremonies   │
│ MINOR - New capabilities that preserve existing contracts   │
│ PATCH - Corrections to unintended behaviors                 │
│ LABEL - Development stage markers for the adventurous       │
│ BUILD - Implementation identifiers for perfect traceability │
└─────────────────────────────────────────────────────────────┘
```

## 🤝 **Contribution Guidelines** 🌱

Contributions must adhere to Eidosian principles:

- **Mathematical Precision** — Version logic must be deterministic and provable
- **Type Safety** — All operations must enforce compile-time guarantees
- **Backward Compatibility** — API changes require explicit versioning
- **Documentation** — All behaviors must be precisely specified
- **Testing** — 100% coverage of compatibility logic

```ascii
╭──────────────────────────────────────────────────────────╮
│ (￣ ω ￣) "In version_forge, we don't estimate           │
│           compatibility—we compute it with certainty."    │
╰──────────────────────────────────────────────────────────╯
```

## 📚 **Further Resources** 📖

- [Semantic Versioning Specification](https://semver.org/) - The mathematical foundation
- [Documentation](https://github.com/Ace1928/version_forge/docs) - Complete reference
- [Compatibility Calculator](https://version-forge.eidosian.dev) - Interactive testing tool
- [Upgrade Path Visualizer](https://version-forge.eidosian.dev/paths) - Migration mapping

---

Maintained with recursive precision by Lloyd Handyside <ace1928@gmail.com>
© 3.14.15 - The irrational version for rational minds

> "A lie in a version number causes an integer overflow in the cognitive error counter." — Eidos
