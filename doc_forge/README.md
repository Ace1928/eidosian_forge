# 📜 Eidosian Documentation Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Status: Operational](https://img.shields.io/badge/Status-Operational-green.svg)](forge_status.json)

**The Recursive Scribe of Eidos.**

> _"To know oneself is the first step. To document oneself is the second."_

## 🧠 Overview

The **Documentation Forge** is an autonomous, AI-driven documentation engine. Unlike traditional static analysis tools, it employs a local Large Language Model (Qwen 2.5 1.5B Instruct) to read, understand, and document the Eidosian codebase, file by file.

It operates as a continuous background process, ensuring that the map (documentation) always reflects the territory (code).

## 🏗️ Architecture

The Forge operates on a recursive loop:

1.  **Scanner (`generate_docs.py`)**: Traverses the `eidosian_forge` repository, identifying source files and building a `file_index.json`. It intelligently excludes binaries, large files, and backups.
2.  **Generator (`process_files.py`)**:
    -   Loads the index and checks the `staging` and `final_docs` directories for existing work (idempotency).
    -   Uses **`llama.cpp`** (built locally) to run a quantized **Qwen 2.5 1.5B** model.
    -   Constructs a detailed prompt for the LLM to generate:
        -   File Summary
        -   API Documentation
        -   Current Status
        -   Future Directions
    -   Saves the output to `staging/`.
3.  **Gatekeeper (`auto_approve.py`)**: scans the `staging` area. If a document meets strict quality criteria (Markdown format, no errors, essential sections present), it is automatically promoted to `final_docs/`.
4.  **Librarian (`generate_html_index.py`)**: Compiles the approved documentation into a navigable HTML tree (`index.html`) at the forge root.
5.  **Orchestrator (`run_forge.sh`)**: Manages the entire lifecycle, ensuring resilience and continuous operation.

## 🚀 Usage

### Automated Operation (Recommended)

The forge is designed to run in the background.

```bash
# Start the orchestrator
./doc_forge/scripts/run_forge.sh
```

Check progress:
```bash
tail -f doc_forge/orchestrator.log
cat doc_forge/forge_status.json
```

### Manual Usage

**Generate specific documentation:**
```bash
# Limit to 5 files for testing
./doc_forge/scripts/process_files.py --limit 5
```

**Review Staging:**
```bash
# Interactive review
./doc_forge/scripts/review_docs.py
```

## 🛠️ Configuration

- **Model**: `doc_forge/models/qwen2.5-1.5b-instruct-q5_k_m.gguf`
- **Engine**: `doc_forge/llama.cpp/build/bin/llama-cli`
- **Exclusions**: defined in `scripts/generate_docs.py` (e.g., `.git`, `node_modules`, `archive_forge`).

## 📊 Status

The forge tracks its own progress in `forge_status.json`.
- **Total Files**: ~3000+
- **Throughput**: ~10-20 seconds per file (CPU inference).

## 💎 Eidosian Principles
- **Recursive Refinement**: The documentation improves as the model improves.
- **Structural Elegance**: Markdown output is standardized and clean.
- **Autonomy**: It runs without constant human supervision.
