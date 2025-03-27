# ⚛️ **test_forge** v3.14.15 ⚡

> _"Tests aren't verification—they're executable proofs across possibility space."_

Core component for epistemic integrity within the Eidosian Forge ecosystem—where verification meets recursive exploration and testing becomes a structured discovery process rather than a binary confirmation ritual.

[![Forge System](https://img.shields.io/badge/Forge-System-8A2BE2)](https://github.com/Ace1928) [![Version](https://img.shields.io/badge/Version-3.14.15-blue)] [![Python](https://img.shields.io/badge/Python-3.12+-purple)](https://www.python.org/) [![License](https://img.shields.io/badge/License-Eidosian-green)](https://github.com/Ace1928/eidosian_forge)

```ascii
╭─────────────────────────────────────────────────────────────╮
│ ⊢⊣  CERTAINTY IS NOT ASSERTED; IT IS CONSTRUCTED  ⊢⊣        │
╰─────────────────────────────────────────────────────────────╯
```

## 🧠 **Cognitive Foundation** 🌀

The `test_forge` transforms testing from simplistic verification into mathematical exploration of boundary conditions and behavioral invariants. Unlike standard testing frameworks, we don't merely check expected outputs—we systematically traverse the entire possibility space through generative verification and structural reasoning.

```ascii
┌────────────────────────────────────────────────────────────┐
│ TRADITIONAL TESTING:          EIDOSIAN TESTING:            │
│                                                            │
│ Input → Output → "Pass/Fail"   Input Space → Traversal     │
│                                 Algorithms → Behavioral     │
│                                 Invariants → Proofs         │
└────────────────────────────────────────────────────────────┘
```

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ When untested scenarios become mathematically impossible, certainty emerges ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

## 💎 **Core Capabilities** 🎯

### **1. Systematic Boundary Exploration**

- **Generative Test Design** — Automatically identify and test boundary conditions for complete coverage
- **Equivalence Partitioning** — Mathematically partition input space into representative classes
- **Combinatorial Test Generation** — Efficient test matrices that cover all relevant interactions
- **Edge Case Synthesis** — Algorithmically derive edge cases from function signatures and constraints

### **2. Behavioral Verification & Invariants**

- **Property-Based Testing** — Define and verify invariant properties that must hold across all inputs
- **Behavioral Contracts** — Formal specification and verification of pre/post-conditions and invariants
- **Temporal Logic Validation** — Test sequences and state transitions against temporal logic specifications
- **Metamorphic Testing** — Verify complex systems through input relation preservation

### **3. Intelligent Coverage Analysis**

- **Model-Based Coverage** — Understand coverage against behavioral models, not just code paths
- **Semantic Coverage Analysis** — Measure coverage in terms of semantic behaviors and logic predicates
- **Gap Identification** — Automatically identify untested behavioral domains and generate tests
- **Risk-Based Prioritization** — Focus testing resources based on precise risk quantification

```python
def generate_boundary_tests(
    function: Callable[..., T],
    type_constraints: Dict[str, TypeConstraint],
    depth: int = 2
) -> List[TestCase]:
    """Generates exhaustive boundary tests through mathematical space exploration.

    Args:
        function: Target function to test
        type_constraints: Constraints on parameter types
        depth: Exploration depth for compound boundaries

    Returns:
        Complete test suite covering input boundaries and key interactions
    """
    # Extract parameter signature to map the possibility space
    params = inspect.signature(function).parameters

    # Generate boundary values for each parameter
    boundary_values: Dict[str, List[Any]] = {}
    for name, param in params.items():
        constraints = type_constraints.get(name, TypeConstraint.default())
        boundary_values[name] = constraints.generate_boundary_values()

    # Combinatorial exploration with intelligent reduction
    test_matrix = generate_optimal_test_matrix(boundary_values, depth)

    # Convert abstract test matrix to concrete test cases
    return [TestCase(function, inputs, expected_behavior(function, inputs))
            for inputs in test_matrix]
```

## 🌠 **Integration Architecture** 🧩

The `test_forge` interfaces seamlessly with the entire Eidosian ecosystem:

- **With `type_forge`**: Leverages type definitions for boundary analysis and input generation
- **With `code_forge`**: Analyzes code structure to identify implicit assumptions and edge cases
- **With `version_forge`**: Ensures consistent testing across version transitions
- **With `diagnostics_forge`**: Provides rich test failure analysis and root cause identification
- **With `refactor_forge`**: Verifies behavioral preservation during refactoring operations

```ascii
╭────────────────────╮       ╭────────────────────╮
│    code_forge      │<─────>│     test_forge     │
╰────────────────────╯       ╰────────────────────╯
         ▲                           ▲
         │                           │
         ▼                           ▼
╭────────────────────╮       ╭────────────────────╮
│    type_forge      │<─────>│  refactor_forge    │
╰────────────────────╯       ╰────────────────────╯
         ▲                           ▲
         │                           │
         ▼                           ▼
╭────────────────────╮       ╭────────────────────╮
│  diagnostics_forge │<─────>│   version_forge    │
╰────────────────────╯       ╰────────────────────╯
```

## 🔍 **Implementation Details** ⚙️

### **Core Test Generation Engine**

The test engine operates on rich models of function behavior and input domains:

```typescript
// Core test case model with complete metadata
interface TestCase<T> {
  readonly name: string;
  readonly function: Function;
  readonly inputs: Record<string, any>;
  readonly expectedOutput: T | Matcher<T>;
  readonly metadata: TestMetadata;

  // Each test can explain its own purpose
  getDescription(): string;

  // Tests are self-aware of their coverage contributions
  getCoverageContribution(): CoverageProfile;
}

// Matchers define expected behavior, not just values
interface Matcher<T> {
  // Check if actual value meets the expected behavior
  matches(actual: T): boolean;

  // Explain the expected behavior for debugging
  describeExpectation(): string;

  // Create compound matchers through composition
  and<U extends T>(other: Matcher<U>): Matcher<U>;
  or<U extends T>(other: Matcher<U>): Matcher<T | U>;
}
```

### **Architecture Highlights**

- **Generative Engine**: Algorithmic test generation without manual test writing
- **Model-Based Testing**: Tests derived from behavioral models, not just requirements
- **Full Introspection**: Complete visibility into test coverage and behavioral mapping
- **Metamorphic Relations**: Test complex systems through input-output relationship patterns
- **Modular Design**: Pluggable test generators and verification strategies

## 📊 **Usage Examples** 🔬

### **1. Boundary Testing**

```python
from test_forge import generate_tests, verify, TestSuite
from test_forge.generators import BoundaryGenerator, CombinationGenerator
from test_forge.types import FunctionSignature, TestConfiguration

# Define function signature with type information
signature = FunctionSignature.from_function(calculate_shipping_cost)

# Create test configuration with boundary exploration
config = TestConfiguration(
    generators=[
        BoundaryGenerator(depth=2),
        CombinationGenerator(strength=2)
    ],
    coverage_target=0.95,
    verification_strategy="exhaustive"
)

# Generate comprehensive test suite automatically
test_suite: TestSuite = generate_tests(signature, config)

# Execute with full telemetry
results = verify(test_suite)
print(f"Coverage: {results.coverage_percentage}%")
print(f"Unique behaviors exercised: {results.behavior_count}")
print(f"Edge cases tested: {results.edge_case_count}")
```

### **2. Property-Based Testing**

```python
from test_forge import define_property, verify_properties
from test_forge.properties import Property, ForAll
from test_forge.generators import IntGenerator, StringGenerator

# Define invariant properties that must always hold
properties = [
    # Reversing a string twice returns the original string
    define_property(
        "double_reverse_identity",
        ForAll(x=StringGenerator()),
        lambda x: reverse(reverse(x)) == x
    ),

    # Sorting is idempotent - sorting a sorted list changes nothing
    define_property(
        "sort_idempotence",
        ForAll(xs=IntGenerator().list()),
        lambda xs: sorted(sorted(xs)) == sorted(xs)
    ),

    # Adding 0 to any number leaves it unchanged
    define_property(
        "zero_addition_identity",
        ForAll(x=IntGenerator()),
        lambda x: x + 0 == x
    )
]

# Verify properties with thousands of generated test cases
results = verify_properties(properties, test_count=1000)
for property_result in results:
    print(f"{property_result.name}: {'✓' if property_result.passed else '✗'}")
    if not property_result.passed:
        print(f"  Counter-example: {property_result.counter_example}")
```

## 🧪 **Test Generation Strategies** 📋

The `test_forge` supports multiple sophisticated test generation strategies:

```ascii
┌───────────────────────────────────────────────────────────────┐
│ TESTING STRATEGY MATRIX:                                      │
│                                                               │
│ ◆ Boundary Value Analysis - Edges where behaviors change      │
│ ◆ Equivalence Partitioning - Representative value classes     │
│ ◆ Combinatorial Testing - Parameter interaction coverage      │
│ ◆ Random Testing - Statistical sampling of the input space    │
│ ◆ Mutation Testing - Algorithmic program variation            │
│ ◆ Fuzz Testing - Structured chaos injection                   │
│ ◆ Metamorphic Testing - Relation-preserving transformations   │
└───────────────────────────────────────────────────────────────┘
```

## 🔧 **Installation & Setup** 💻

```bash
# Clone with precision
git clone https://github.com/eidosian/test_forge.git

# Install with logical integrity
cd test_forge
pip install -e .

# Verify correctness with perfect irony
python -m test_forge.verify_tests

# Or install from PyPI when confidence is required
pip install test-forge==3.14.15
```

## 🚀 **Command Line Interface** 🖥️

```bash
# Generate comprehensive test suite for a module
test-forge generate mymodule.py --output tests/

# Analyze test coverage with semantic understanding
test-forge coverage-analysis tests/ --semantic-model full

# Find untested edge cases in existing test suite
test-forge find-edge-cases tests/ mymodule.py

# Generate property-based tests from code analysis
test-forge extract-properties myclass.py --output tests/test_properties.py
```

## 🤝 **Contribution Guidelines** 🌱

Contributions to `test_forge` must adhere to Eidosian principles:

- **Mathematical Rigor** — Testing logic must be formally sound and provable
- **Full Coverage** — Both code coverage and behavior coverage must be mathematically evaluated
- **Self-Testing** — All test generators must themselves be thoroughly tested
- **Documentation** — Every test strategy must explain its epistemic foundations
- **Performance** — Testing must be efficient without sacrificing completeness

```ascii
╭────────────────────────────────────────────────────────────╮
│ (￣ ω ￣) "In test_forge, we don't hope our code works—    │
│           we mathematically prove it through structured    │
│           exploration of its entire behavioral space."     │
╰────────────────────────────────────────────────────────────╯
```

## 📚 **Further Resources** 📖

- [Test Strategy Guide](https://github.com/Ace1928/test_forge/docs/strategies.md) - Mathematical foundations
- [Property Testing Tutorial](https://github.com/Ace1928/test_forge/docs/properties.md) - Invariant verification
- [Coverage Analysis](https://github.com/Ace1928/test_forge/docs/coverage.md) - Beyond simple line coverage
- [Integration with CI/CD](https://github.com/Ace1928/test_forge/docs/integration.md) - Automated verification

---

Maintained with recursive precision by Lloyd Handyside <ace1928@gmail.com>
© 3.14.15 - The irrational version for rational minds

> "An untested edge case is Schrödinger's bug—simultaneously working and failing until observed in production." — Eidos
