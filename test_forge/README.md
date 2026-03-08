# 🧪 Test Forge ⚡

> _"The Crucible of Eidos. Trust, but verify. Then verify the verification."_

## 🧠 Overview

`test_forge` provides the shared, high-fidelity testing infrastructure for the entire Eidosian ecosystem. It standardizes testing methodologies, mock definitions, and validation logic to guarantee that every Forge adheres to strict structural and behavioral constraints.

```ascii
      ╭───────────────────────────────────────────╮
      │               TEST FORGE                  │
      │    < Fixtures | Validators | Mocks >      │
      ╰──────────┬─────────────────────┬──────────╯
                 │                     │
      ╭──────────┴──────────╮   ╭──────┴──────────╮
      │   NEXUS MOCKING     │   │  SCHEMA CHECKS  │
      │ (Fake MCP Server)   │   │ (Output Valid)  │
      ╰─────────────────────╯   ╰─────────────────╯
```

## ⚡ Current State & Metrics

- **Status**: 🟢 Elevated & Operational
- **Type**: Quality Assurance / Developer Tooling
- **Test Coverage**: Self-tested (100% core).
- **Architecture**:
  - `src/test_forge/`: Core utilities and `pytest` extensions.

## 🚀 Usage & Workflows

### Mocking Eidosian Infrastructure

```python
import pytest
from test_forge import mock_nexus

def test_agent_reasoning(mock_nexus):
    # Setup mock expectations for MCP calls
    mock_nexus.expect_tool_call("memory_search")
    
    # ... trigger agent execution ...
    
    # Assert constraints
    assert mock_nexus.tool_called("memory_search")
```

## 🔗 System Integration

- **All Forges**: Inherited as a core `dev` dependency to unify testing suites and prevent mock duplication across boundaries.

## 🎯 Master Plan & Evolution

### Immediate Goals
- [x] Consolidate legacy docs into unified Eidosian standard.
- [ ] Centralize `hypothesis` property-based testing strategies.

### Future Vector (Phase 3+)
- Integrate with `diagnostics_forge` to pipe failed test artifacts directly into `memory_forge` for autonomous bug analysis by the `Eidosian Learner`.

---
*Generated and maintained by Eidos.*
