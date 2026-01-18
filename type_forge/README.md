# ⚛️ **type_forge** v3.14.15 ⚡

> _"Types aren't constraints—they're mathematical guarantees across execution space."_

Core component for structural integrity within the Eidosian Forge ecosystem—where validation meets recursive precision and type safety becomes a compile-time certainty rather than a runtime hope.

[![Forge System](https://img.shields.io/badge/Forge-System-8A2BE2)](https://github.com/Ace1928) [![Version](https://img.shields.io/badge/Version-3.14.15-blue)] [![Python](https://img.shields.io/badge/Python-3.12+-purple)](https://www.python.org/) [![License](https://img.shields.io/badge/License-Eidosian-green)](https://github.com/Ace1928/eidosian_forge)

```ascii
╭───────────────────────────────────────────────────────────╮
│ ⊢⊣ TYPE CORRECTNESS IS NOT VERIFIED; IT IS GUARANTEED ⊢⊣  │
╰───────────────────────────────────────────────────────────╯
```

## 🧠 **Cognitive Foundation** 🌀

The `type_forge` transforms type uncertainty into mathematical certainty through recursive validation and structural verification. Unlike standard type systems, we don't merely annotate and hope—we enforce structural coherence through compile-time guarantees and runtime validation.

```ascii
┌────────────────────────────────────────────────────────────┐
│ TRADITIONAL TYPE SYSTEMS:      EIDOSIAN TYPE FORGE:        │
│                                                            │
│ Types → Annotations → Runtime   Types → Executable         │
│ Errors → "Oops!"                Contracts → Compile-time   │
│                                 Certainty                  │
└────────────────────────────────────────────────────────────┘
```

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ When undefined behavior is mathematically impossible, code achieves freedom ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

## 💎 **Core Capabilities** 🎯

### **1. Dynamic Type Creation & Manipulation**

- **Type Construction** — Create, register, and instantiate type definitions with precision guarantees
- **Structural Validation** — Enforce type constraints across arbitrary nested structures
- **Path-Specific Reporting** — Receive mathematically precise error reports at exact violation coordinates
- **Type Algebra** — Compose types through unions, intersections, and differentiations

### **2. Recursive Type Validation**

- **Deep Structure Verification** — Validate complex nested hierarchies with zero escape hatches
- **Constraint Propagation** — Apply validation rules consistently through nested structures
- **Incremental Validation** — Validate partial structures during construction for early feedback
- **Validation Caching** — Optimize validation through intelligent structural fingerprinting

### **3. Type Transformation & Analysis**

- **Type Conversion** — Transform between compatible types with perfect fidelity
- **Compatibility Determination** — Compute precise compatibility relationships between types
- **Structural Equivalence** — Determine when types are isomorphic regardless of naming
- **Type Registry** — Manage types in isolated namespaces with conflict prevention

```python
def validate_recursive_structure(
    structure: Any,
    type_schema: TypeDefinition,
    path: List[str] = None
) -> ValidationResult:
    """Validates complex nested structures with mathematical precision.

    Args:
        structure: The data to validate against the schema
        type_schema: The type definition to validate against
        path: Current path for error reporting

    Returns:
        Complete validation result with exact violation coordinates
    """
    path = path or []

    # Type match is a mathematical proof, not a hopeful check
    if not isinstance(structure, type_schema.expected_type):
        return ValidationResult.failure(
            f"Expected {type_schema.expected_type.__name__} at {'.'.join(path)}"
        )

    # Recursive validation—types all the way down
    for field_name, field_schema in type_schema.fields.items():
        field_path = path + [field_name]

        # Structure must be complete—partial truths are falsehoods
        if field_name not in structure:
            return ValidationResult.failure(
                f"Missing required field {'.'.join(field_path)}"
            )

        # Recursion proceeds through the entire structure
        field_result = validate_recursive_structure(
            structure[field_name], field_schema, field_path
        )

        # Invalid subtrees invalidate the whole tree
        if not field_result.valid:
            return field_result

    # A fully validated structure is a mathematical certainty
    return ValidationResult.success()
```

## 🌠 **Integration Architecture** 🧩

The `type_forge` sits at the heart of the Eidosian ecosystem, providing structural guarantees to all components:

- **With `version_forge`**: Schema versioning with compatibility verification
- **With `repo_forge`**: Type-safe repository structure validation
- **With `gis_forge`**: Configuration schema definition and validation
- **With `diagnostics_forge`**: Typed error hierarchies and structured logging
- **With `knowledge_forge`**: Schema-validated knowledge representation

```ascii
╭────────────────────╮     ╭────────────────────╮
│  All Components    │ → → │    type_forge      │
╰────────────────────╯     ╰────────────────────╯
                                     ↓
╭────────────────────╮     ╭────────────────────╮
│  Python Runtime    │ ← ← │  Type Guarantees   │
╰────────────────────╯     ╰────────────────────╯
```

## 🔍 **Implementation Details** ⚙️

### **Core Type System**

The type system provides a rich domain model for structural validation:

```typescript
// Core type definition with recursive structure
interface TypeDefinition {
  readonly name: string;
  readonly expectedType: Type;
  readonly fields: Record<string, TypeDefinition>;
  readonly constraints: TypeConstraint[];
  readonly metadata: TypeMetadata;

  // Validation is mathematically precise
  validate(value: any, path?: string[]): ValidationResult;

  // Types can be composed algebraically
  union(other: TypeDefinition): TypeDefinition;
  intersection(other: TypeDefinition): TypeDefinition;
  withField(name: string, type: TypeDefinition): TypeDefinition;
}

// Validation results contain exact violation coordinates
interface ValidationResult {
  readonly valid: boolean;
  readonly errors: ValidationError[];
  readonly path: string[];

  // Results can be aggregated across validation runs
  concat(other: ValidationResult): ValidationResult;
}
```

### **Architecture Highlights**

- **Zero Any Types**: Complete type safety with no escape hatches
- **Immutable Type Definitions**: Type definitions are unchangeable after creation
- **Optimized Validation**: Validation with structural fingerprinting for performance
- **Extensible Constraint System**: Custom constraints through composable interfaces
- **API Surface Minimalism**: Small, elegant API with maximum expressive power

## 📊 **Usage Examples** 🔬

### **1. Type Definition & Validation**

```python
from type_forge import define_type, validate, TypeDefinition
from type_forge.constraints import Required, Min, Max, Pattern

# Define types with precise constraints
user_type: TypeDefinition = define_type("User", {
    "id": define_type("UserId", str).with_constraints(
        Pattern(r"^[a-z0-9]{8,12}$")
    ),
    "name": define_type("UserName", str).with_constraints(
        Required(), Min(2), Max(100)
    ),
    "age": define_type("UserAge", int).with_constraints(
        Min(0), Max(150)
    ),
    "email": define_type("Email", str).with_constraints(
        Pattern(r"^[\w.-]+@[\w.-]+\.\w+$")
    )
})

# Validate with path-specific error reporting
valid_user = {
    "id": "abc123def456",
    "name": "Alice Wonderland",
    "age": 30,
    "email": "alice@wonderland.com"
}

result = validate(valid_user, user_type)
print(result.valid)  # True

# Invalid data receives precise error coordinates
invalid_user = {
    "id": "SHORT",
    "name": "Bob",
    "age": 200,  # Exceeds maximum
    "email": "not-an-email"
}

result = validate(invalid_user, user_type)
print(result.valid)  # False
print(result.errors)  # [ValidationError at 'id': Pattern mismatch, ValidationError at 'age': Maximum exceeded...]
```

### **2. Dynamic Type Creation & Transformation**

```python
from type_forge import define_type, transform, TypeRegistry

# Create a type registry with isolated namespaces
registry = TypeRegistry()

# Register types for reuse
registry.register("coordinates", define_type("Coordinates", {
    "x": define_type("X", float),
    "y": define_type("Y", float),
    "z": define_type("Z", float, required=False)
}))

# Create a new type that extends an existing one
point_type = registry.get("coordinates").extend({
    "label": define_type("Label", str),
    "color": define_type("Color", str, required=False)
})

# Transform between compatible types
geo_point = {
    "latitude": 37.7749,
    "longitude": -122.4194,
    "name": "San Francisco"
}

# Define transformation rules
transformation_rules = {
    "latitude": "y",
    "longitude": "x",
    "name": "label"
}

# Transform with perfect fidelity
point = transform(geo_point, point_type, transformation_rules)
print(point)  # {"x": -122.4194, "y": 37.7749, "label": "San Francisco"}
```

## 🔧 **Installation & Setup** 💻

```bash
# Install from PyPI with mathematical certainty
pip install type-forge==3.14.15

# Verify installation integrity
python -m type_forge.verify

# Or install from source
git clone https://github.com/eidosian/type_forge.git
cd type_forge
pip install -e .
```

## 🚀 **Command Line Interface** 🖥️

```bash
# Validate data files against schemas
type-forge validate data.json schema.json

# Generate type definitions from data samples
type-forge generate-schema samples.json --output schema.json

# Check type compatibility across components
type-forge compatibility-check user_v1.json user_v2.json

# Analyze type usage across a codebase
type-forge analyze-types ./src --report report.html
```

## 🤝 **Contribution Guidelines** 🌱

Contributions to `type_forge` must adhere to Eidosian principles:

- **Zero Any Types** — No escape hatches or type uncertainty
- **Full Test Coverage** — Every type operation must be mathematically verified
- **API Minimalism** — Maximum power through minimal, composable interfaces
- **Documentation Precision** — Every function, parameter, and return type fully documented
- **Performance Awareness** — O-notation specified for all validation operations

```ascii
╭──────────────────────────────────────────────────────╮
│ (￣ ω ￣) "In type_forge, we don't check types;     │
│           we mathematically guarantee them."         │
╰──────────────────────────────────────────────────────╯
```

## 📚 **Further Resources** 📖

- [Getting Started Guide](https://github.com/Ace1928/type_forge/docs/getting_started.md)
- [API Reference](https://github.com/Ace1928/type_forge/docs/api/index.md)
- [Type Constraint System](https://github.com/Ace1928/type_forge/docs/api/constraints.md)
- [Integration with Other Forges](https://github.com/Ace1928/type_forge/docs/integration.md)

---

Maintained with recursive precision by Lloyd Handyside <ace1928@gmail.com>
© 3.14.15 - The irrational version for rational minds

> "Type correctness isn't a matter of hope—it's a matter of mathematical proof." — Eidos
