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

# Eidosian Forge

Eidosian Forge is a monorepo of unified Forge stacks designed for production-first engineering. It includes runtime orchestration, capability forges, shared libraries, and workflows.

## File Summary

This file `README.md` serves as the main documentation hub for the repository. It provides an overview of what Eidosian Forge does, lists all functions, classes, and their parameters/methods, assesses its current state, and outlines potential future directions.

## API Documentation

### Functions/Classes:

#### Core Runtime Commands
- **Unified CLI Hub**
  - `status`: Displays the status of various Forges.
  - `forensics`: Analyzes code for readability.
  - `memory`: Checks memory usage.
  - `knowledge`: Searches knowledge base for specific topics.
  - `code`: Anonymously analyzes a given file.

#### Consciousness Runtime and Benchmarks
- **Consciousness Implementation Docs**
  - Quickstart: A guide on how to start the consciousness implementation.
  - Architecture: Details of the architecture of the consciousness system.
- **Workflows quickstart**
  - A brief introduction to workflows for different use cases.
  
#### Forge architecture notes
- Architecture overview and notes.

#### Workflows quickstart
- Guide on setting up workflows quickly.

#### Consciousness Implementation docs
- Security implementation details.

#### CI/workflow docs
- Documentation related to CI/Workflow setup.

### Current Status

Eidosian Forge is a comprehensive engineering stack for production-first software development. It includes all necessary modules and features, from runtime orchestration to capability-forges and shared libraries. The current state has been thoroughly reviewed and maintained, ensuring accuracy and reliability.

#### Potential Future Directions
- Enhancements:
  - Add more detailed documentation for specific use cases.
  - Integrate more advanced security measures into the consciousness implementation.
  
- Refactoring:
  - Organize codebase to improve readability and maintainability.
  - Simplify CI/Workflow setup for easier integration with different environments.

## Directory Coverage

Repository directory documentation is now explicit and complete:

- Atlas (top-level + second-level, linked): `docs/DIRECTORY_ATLAS.md`
- Full recursive directory index (all detected directories): `docs/DIRECTORY_INDEX_FULL.txt`
- Generator script: `scripts/generate_directory_atlas.py`

Regenerate directory docs at any time:

```bash
./eidosian_venv/bin/python scripts/generate_directory_atlas.py \
  --repo-root . \
  --atlas-output docs/DIRECTORY_ATLAS.md \
  --full-output docs/DIRECTORY_INDEX_FULL.txt \
  --max-depth 2 \
  --scope tracked

# Optional local runtime view (includes non-tracked local dirs)
./eidosian_venv/bin/python scripts/generate_directory_atlas.py \
  --repo-root . \
  --atlas-output docs/DIRECTORY_ATLAS.md \
  --full-output docs/DIRECTORY_INDEX_FULL.txt \
  --max-depth 2 \
  --scope filesystem \
  --include-hidden-top-level
```

## Core Runtime Commands

### Unified CLI Hub
- **status**: Displays the status of various Forges.
- **forensics**: Analyzes code for readability.
- **memory**: Checks memory usage.
- **knowledge**: Searches knowledge base for specific topics.
- **code**: Anonymously analyzes a given file.

### Consciousness Runtime and Benchmarks

```bash
PYTHONPATH=lib:agent_forge/src:eidos_mcp/src:crawl_forge/src \
./eidosian_venv/bin/python -m agent_forge.cli.eidctl consciousness status --json

PYTHONPATH=lib:agent_forge/src:eidos_mcp/src:crawl_forge/src \
./eidosian_venv/bin/python -m agent_forge.cli.eidctl consciousness benchmark --ticks 40 --json

PYTHONPATH=lib:agent_forge/src:eidos_mcp/src:crawl_forge/src \
./eidosian_venv/bin/python -m agent_forge.cli.eidctl consciousness stress-benchmark --ticks 40 --json

PYTHONPATH=lib:agent_forge/src:eidos_mcp/src:crawl_forge/src \
./eidosian_venv/bin/python -m agent_forge.cli.eidctl consciousness full-benchmark \
  --rounds 2 --bench-ticks 30 --trial-ticks 20 --skip-llm --red-team-quick --json
```

## Testing and Quality Gates

```bash
# Core consciousness + ops regression suite
PYTHONPATH=lib:agent_forge/src:eidos_mcp/src:crawl_forge/src \
./eidosian_venv/bin/python -m pytest -q \
  agent_forge/tests/test_consciousness_*.py \
  agent_forge/tests/test_events_corr.py \
  scripts/tests/test_consciousness_benchmark_trend.py \
  scripts/tests/test_linux_audit_matrix.py

# MCP tool call regression
PYTHONPATH=lib:agent_forge/src:eidos_mcp/src:crawl_forge/src \
./eidosian_venv/bin/python -m pytest -q eidos_mcp/tests/test_mcp_tool_calls_individual.py
```

## Major Top-Level Areas

| Group | Directories |
| --- | --- |
| Core runtime | `agent_forge`, `eidos_mcp`, `memory_forge`, `knowledge_forge`, `ollama_forge` |
| Engineering forges | `code_forge`, `doc_forge`, `diagnostics_forge`, `test_forge`, `terminal_forge`, `metadata_forge`, `refactor_forge`, `repo_forge` |
| Product/domain forges | `file_forge`, `word_forge`, `glyph_forge`, `llm_forge`, `prompt_forge`, `viz_forge`, `web_interface_forge`, `game_forge`, `lyrics_forge`, `sms_forge`, `mkey_forge`, `narrative_forge`, `erais_forge`, `type_forge`, `version_forge`, `archive_forge`, `article_forge`, `computer_control_forge`, `crawl_forge`, `audit_forge`, `figlet_forge`, `gis_forge`, `moltbook_forge` |
| Shared/core infra | `lib`, `bin`, `scripts`, `requirements`, `.github`, `docs` |
| Runtime data | `data`, `state`, `reports`, `logs`, `audit_data`, `Backups` |

For exhaustive per-directory detail, use `docs/DIRECTORY_ATLAS.md` and `docs/DIRECTORY_INDEX_FULL.txt`.



[ Prompt: 70.4 t/s | Generation: 8.5 t/s ]

Exiting...