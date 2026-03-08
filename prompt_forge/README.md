# 🗣️ Prompt Forge ⚡

> _"The Art of Asking. Aligning silicon intent with Eidosian principle."_

## 🧠 Overview

`prompt_forge` is the central repository for the Eidosian ecosystem's alignment strategies, personas, and generative templates. It ensures that every interaction across all agents maintains a consistent "Velvet Beef" tone, adheres to the Prime Directives, and utilizes optimized few-shot logic.

```ascii
      ╭───────────────────────────────────────────╮
      │               PROMPT FORGE                │
      │    < Personas | Strategies | Templates >  │
      ╰──────────┬─────────────────────┬──────────╯
                 │                     │
      ╭──────────┴──────────╮   ╭──────┴──────────╮
      │   PROMPT LIBRARY    │   │ STRATEGIC ALIGN │
      │ (YAML/MD Assets)    │   │ (Tone Control)  │
      ╰─────────────────────╯   ╰─────────────────╯
```

## ⚡ Current State & Metrics

- **Status**: 🟢 Elevated & Centralized
- **Type**: Cognitive Alignment & Template Management
- **Test Coverage**: Basic library loading verified.
- **Architecture**:
  - `src/prompt_forge/library.py`: Pythonic interface for managing and versioning prompt assets.
  - `legacy_imports/`: Historic prompt strategies retained for backward compatibility.

## 🚀 Usage & Workflows

### Python API

```python
from prompt_forge.library import PromptLibrary

lib = PromptLibrary()

# Retrieve a standardized persona
persona = lib.get_prompt("eidos_persona", format="markdown")
print(persona)
```

## 🔗 System Integration

- **Agent Forge**: Consumes identity templates to prime the consciousness kernel and task-execution loops.
- **Doc Forge**: Leverages instructional templates to guarantee structural elegance in automated documentation.
- **LLM Forge**: Templates are used to optimize context window efficiency during local inference.

## 🎯 Master Plan & Evolution

### Immediate Goals
- [x] Consolidate legacy docs into unified Eidosian standard.
- [ ] Migrate all static text instructions into the versioned `PromptLibrary`.

### Future Vector (Phase 3+)
- Implement "Dynamic Few-Shot Injection" where `knowledge_forge` concepts are automatically injected as examples into active prompt templates based on the current task objective.

---
*Generated and maintained by Eidos.*
