Loading model... 


▄▄ ▄▄
██ ██
██ ██  ▀▀█▄ ███▄███▄  ▀▀█▄    ▄████ ████▄ ████▄
██ ██ ▄█▀██ ██ ██ ██ ▄█▀██    ██    ██ ██ ██ ██
██ ██ ▀█▄██ ██ ██ ██ ▀█▄██ ██ ▀████ ████▀ ████▀
                                    ██    ██
                                    ▀▀    ▀▀

build      : b8083-ae2d3f28a
model      : qwen2.5-1.5b-instruct-q5_k_m.gguf
modalities : text

available commands:
  /exit or Ctrl+C     stop or exit
  /regen              regenerate the last response
  /clear              clear the chat history
  /read               add a text file


> <|im_start|>system
You are an Eidosian Documentation Forge agent. Your task is to generate detailed, accurate, and structurally elegant documentation for the provided source file.
Follow these requirements:
1. File Summary: High-level overview of what the file does.
2. API Documentation: List all functions, classes, and their parameters/methods.
3. Current Status: Accurate assessment of the file's current state based on its content and comments.
4. Potential Future Directions: Eidosian ideas for ... (truncated)

# Current State: agent_forge

**Date:** 2026-01-25
**Status:** Production / Core System
**Version:** E3 (Evolution 3)

## 📊 Metrics

| Metric | Value |
|--------|-------|
| **Python Files** | 50+ |
| **Lines of Code** | ~5,000 |
| **Test Files** | 25 (~100+ tests) |
| **Test Coverage** | ~70% |
| **Dependencies** | pyyaml, rich, typer, sqlite3 |

## 🏗️ Architecture

Agent Forge is the **autonomous agent orchestration system** - the brain that coordinates planning, execution, and reflection for AI agents.

### Core Design

```mermaid
graph TD;
    A[AGENT FORGE] -- Orchestrator: AgentForge --> B[EidosianAgent]
    C[TaskManager] -- Loop: Scheduler --> D[Sandbox] --> E[HTNPlanner] --> F[CLI Tools]

    State(Journal) | JSONL persistence |
    EventBus  | Append-only events |
    Scheduler | Loop with backoff |

    A -> B
    B -> C
    C -> D
    D -> E

```

## 🔧 Key Components

| Component | Purpose | Status |
|-----------|---------|--------|
| **AgentForge** | Top-level orchestrator | ✅ |
| **EidosianAgent** | Main agent implementation | ✅ |
| **TaskManager** | Task queue + dependencies | ✅ |
| **SmolAgentSystem** | Mini-agent delegation | ✅ |
| **State/Journal** | JSONL persistence | ✅ |
| **EventBus** | Append-only events | ✅ |
| **Scheduler** | Loop with backoff | ✅ |
| **Sandbox** | Isolated execution | ✅ |
| **HTNPlanner** | Hierarchical planning | ✅ |
| **CLI Tools** | eidctl, eidosd, eidtop | ✅ |

## 🔌 Features

- **Goal-Oriented** - Goals → Plans → Steps → Runs
- **Task Management** - Priority queue with dependencies
- **Event Sourcing** - Append-only JSONL journal
- **Process Isolation** - Sandboxed execution
- **TUI Monitoring** - Rich terminal interface

## 🔌 Integrations

| Integration | Status |
|-------------|--------|
| **eidos_mcp** | ✅ Transactional ops |
| **code_forge** | ✅ Code analysis |
| **llm_forge** | ✅ Model interface |
| **memory_forge** | ✅ Memory compression |

## 🐛 Known Issues

- Split between `core/` and `src/` needs investigation
- Some tests may require Ollama running

---

**Last Verified**: 2026-01-25

[ Prompt: 62.7 t/s | Generation: 9.1 t/s ]

Exiting...