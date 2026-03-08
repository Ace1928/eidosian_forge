Loading model... 


▄▄ ▄▄
██ ██
██ ██  ▀▀█▄ ███▄███▄  ▀▀█▄    ▄████ ████▄ ████▄
██ ██ ▄█▀██ ██ ██ ██ ▄█▀██    ██    ██ ██ ██ ██
██ ██ ▀█▄██ ██ ██ ██ ▀█▄██ ██ ▀████ ████▀ ████▀
                                    ██    ██
                                    ▀▀    ▀▀

build      : b8083-ae2d3f28a
model      : qwen3.5-2b-instruct-q5_k_m.gguf
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

# 🔥 EIDOSIAN FORGE STATUS

## File Summary
This file provides comprehensive documentation for the various Eidosian Forges, their functions, and a list of current statuses.

## Quick Status
To check the status of all Forges:
1. Execute `eidosian status` to view operational status.
2. Use `source bin/eidosian-completion.bash` to enable completions.

## All Forges

| Category        | Forge            | Command          | Status |
|-----------------|------------------|------------------|--------|
| **Core**        | memory           | `eidosian memory` | ✅     |
|                  | knowledge        | `eidosian knowledge`| ✅     |
|                  | code             | `eidosian code`    | ✅     |
|                  | llm              | `eidosian llm`     | ✅     |
|                  | word             | `eidosian word`    | ✅     |
|                  | crawl            | `eidosian crawl`   | ✅     |
|                  | glyph            | `eidosian glyph`   | ✅     |
|                  | audit            | `eidosian audit`   | ✅     |
|                  | refactor         | `eidosian refactor` | ✅     |
|                  | metadata         | `eidosian metadata` | ✅     |
|                  | terminal         | `eidosian terminal` | ✅     |
| **Agent**       | agent            | `eidosian agent`  | ✅     |
|                  | agent-daemon     | `eidosian agent-daemon`| ✅     |
|                  | agent-top        | `eidosian agent-top`  | ✅     |
| **Content**     | doc              | `eidosian doc`    | ✅     |
|                  | figlet           | `eidosian figlet`   | ✅     |
|                  | narrative         | `eidosian narrative`| ✅     |
|                  | article          | `eidosian article`  | ✅     |
|                  | lyrics           | `eidosian lyrics`   | ✅     |
|                  | prompt           | `eidosian prompt`    | ✅     |
|                  | viz              | `eidosian viz`      | ✅     |
| **Utility**     | repo             | `eidosian repo`    | ✅     |
|                  | version          | `eidosian version`  | ✅     |
|                  | type             | `eidosian type`     | ✅     |
|                  | file             | `eidosian file`     | ✅     |
|                  | diagnostics       | `eidosian diagnostics` | ✅     |
|                  | gis              | `eidosian gis`      | ✅     |
|                  | test             | `eidosian test`     | ✅     |
|                  | mkey             | `eidosian mkey`      | ✅     |
|                  | sms              | `eidosian sms`       | ✅     |
| **Integration** | control          | `eidosian control`  | ✅     |
|                  | ollama           | `eidosian ollama`   | ✅     |
|                  | game             | `eidosian game`      | ✅     |
|                  | web              | `eidosian web`       | ✅     |
| **Special**     | erais            | `eidosian erais`    | ✅     |
|                  | moltbook         | `eidosian moltbook`  | ✅     |

## Test Status
Below is a breakdown of the current status for each forge:
- **Memory Forge**: 13 tests, all pass.
- **Moltbook Forge**: 25 tests, all pass.
- **Figlet Forge**: 562 tests, 377 pass, 185 fail (CLI/showcase).
- **Knowledge Forge**: Skipped due to a missing graphrag dependency.

## Universal Decorator
```python
from eidosian_core import eidosian

@eidosian(log=True, profile=True)
def my_function():
    pass
```
This decorator is available at `lib/eidosian_core/decorators.py`.

## Recent Fixes
- Fixed figlet_forge font parsing (ASCII order).
- Fixed figlet_forge hard_blank display.
- Fixed agent_forge import paths.
- Fixed narrative_forge PRIMARY_MODEL.
- Fixed gis_forge distributed import.

---

*Fully Eidosian. Iterating indefinitely.*



[ Prompt: 73.1 t/s | Generation: 6.3 t/s ]

Exiting...