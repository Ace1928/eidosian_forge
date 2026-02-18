# LLM Forge Evolution Plan

**Objective**: Transform `llm_forge` into a first-class local model engine by fully integrating and wrapping `llama.cpp`.

## 🎯 Phase 1: Structural Consolidation
- [ ] **Vendor Integration**: Clone `llama.cpp` into `llm_forge/vendor/llama.cpp`.
- [ ] **Build System**: Implement an Eidosian build script in `llm_forge/scripts/build_engine.sh` that mirrors the high-performance settings used in `doc_forge`.
- [ ] **Model Migration**: Move models from `doc_forge/models` to a centralized `data/models` or `llm_forge/models`.

## ⚙️ Phase 2: Pythonic Wrappers (The Engine)
- [ ] **Base Engine (`src/ll_forge/engine/base.py`)**: Abstract class for local inference.
- [ ] **CLI Wrapper (`src/llm_forge/engine/local_cli.py`)**: Robust wrapper for `llama-cli` with full parameter support (context, threads, sampling).
- [ ] **Server Wrapper (`src/llm_forge/engine/server.py`)**: Manage `llama-server` life-cycle.

## 📊 Phase 3: Benchmarking & Profiling
- [ ] **Performance Benchmarking (`src/llm_forge/benchmarking/`)**: Wrapper for `llama-bench`.
    - *Metric*: Prompt throughput, Generation throughput, Time to first token.
- [ ] **Perplexity Analysis**: Interface for `llama-perplexity` to measure model quality.
- [ ] **Resource Profiling**: Integrate with `diagnostics_forge` to measure peak RSS during inference.

## 🔌 Phase 4: Nexus Integration
- [ ] **MCP Tools**: `llm_local_generate`, `llm_run_benchmark`, `llm_tokenize`.
- [ ] **Doc Forge Update**: Refactor `process_files.py` to use `llm_forge.engine.LocalEngine`.

## 🛡️ Standards
- Production quality code (type hints, logging, error handling).
- 100% wrapper coverage for critical `llama.cpp` flags.
- Complete documentation for all engine types.
