# Current State: refactor_forge

**Date**: 2026-01-20
**Status**: Beta

## 📊 Metrics
- **Dependencies**: `libcst`.
- **Files**: `refactor_core.py`, `analyzer.py`.

## 🏗️ Architecture
LibCST-based transformer.

## 🐛 Known Issues
- Directory structure (`libs/`, `src/`) is generic template bloat.
- Duplicate functionality with `code_forge`.
    - **Resolution**: `code_forge` is the *conceptual* domain (High Level). `refactor_forge` is the *implementation* tool (Low Level).