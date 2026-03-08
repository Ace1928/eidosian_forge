# 📖 Narrative Forge ⚡

> _"The Storyteller of Eidos. We are the stories we tell ourselves."_

## 🧠 Overview

`narrative_forge` is the cognitive engine for long-form coherence, world-building, and thematic storytelling. While `agent_forge` handles immediate execution cycles (beats), `narrative_forge` maintains the broader "arc" of an interaction, ensuring logical and atmospheric consistency over extended sessions.

```ascii
      ╭───────────────────────────────────────────╮
      │             NARRATIVE FORGE               │
      │    < Plot | Character | State Machine >   │
      ╰──────────┬─────────────────────┬──────────╯
                 │                     │
      ╭──────────┴──────────╮   ╭──────┴──────────╮
      │  NARRATIVE MEMORY   │   │ THEMATIC ENGINE │
      │ (Context Window)    │   │ (LLM Grounding) │
      ╰─────────────────────╯   ╰─────────────────╯
```

## ⚡ Current State & Metrics

- **Status**: 🟢 Elevated & Operational
- **Type**: Cognitive/Generative Structuring
- **Test Coverage**: Core narrative loop verified.
- **Architecture**:
  - `engine.py`: Manages the narrative state machine and context generation via `llm_forge`.
  - `memory.py`: Dedicated, linear context window for tracking story beats (distinct from semantic graphs).
  - `cli.py`: Interactive interactive storytelling interface.

## 🚀 Usage & Workflows

### Interactive Session

Launch an interactive narrative session powered by the local LLM:
```bash
python -m narrative_forge.cli chat --model "local"
```

### Python API

```python
from narrative_forge.engine import NarrativeEngine

engine = NarrativeEngine(model_name="local")

response = engine.generate_response(
    prompt="You stand before the gates of the Eidosian Citadel.",
    character="The Architect"
)
print(response)
```

## 🔗 System Integration

- **Agent Forge**: Agents query the narrative engine to understand their position in the larger "plot" of a task.
- **Word Forge**: Leverages semantic lexicons to enforce specific thematic vocabularies.
- **Game Forge**: Drives the underlying lore and interaction text for environments like `eidosian_universe`.

## 🎯 Master Plan & Evolution

### Immediate Goals
- [x] Consolidate fragmented legacy documentation.
- [x] Verify local integration loop.

### Future Vector (Phase 3+)
- Integrate natively with `knowledge_forge` to automatically pull in world-building lore as contextual grounding.
- Implement explicit branching narrative trees using a state-graph schema.

---
*Generated and maintained by Eidos.*
