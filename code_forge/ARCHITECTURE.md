# Code Forge Architecture

## Goals

- Ingest source code into a structured, searchable, and updatable library.
- Preserve full reconstruction fidelity (file/module/class/function/statement).
- Provide stable source maps and reversible, idempotent updates.
- Support multiple languages via pluggable parsers (Python first).
- Provide two modes: non-destructive analysis and destructive archival.

## Core Concepts

### Code Unit
A Code Unit is the smallest addressable element in the library. Examples:
- File
- Module
- Class
- Function / Method
- Block (e.g., loop, conditional)
- Statement / Expression

Each unit has a stable ID and a source map.

### Stable ID
Stable IDs are derived from:
- Language
- File path (relative to ingestion root)
- Symbol path (module.class.function)
- Structural signature (AST hash)

This enables diffable and idempotent updates.

### Source Map
Each Code Unit stores:
- file_path
- line_start / line_end
- col_start / col_end
- byte_start / byte_end (optional)
- content_hash

### Library
The library is a normalized store containing:
- code_units table (metadata + pointers)
- code_text table (content blobs, deduped by hash)
- relationships (parent/child, call graph, imports)
- tags / embeddings (optional)

## Ingestion Pipeline

1. **Discovery**: Walk directories and identify source files.
2. **Parsing**: Parse file into AST (Python via `ast`, later via tree-sitter).
3. **Extraction**: Generate Code Units for hierarchical nodes.
4. **Indexing**: Store units + source maps + hashes.
5. **Linking**: Parent/child, imports, references.
6. **Snapshot**: Record ingestion run + configuration.

## Modes

### Analysis & Archival (Non-Destructive)
- Ingest without modifying originals.
- Optional copy of input tree.

### Archival & Consolidation (Destructive)
- Ingest + verify.
- Create compressed snapshot (default).
- Remove originals only after verification and user confirmation.

## Update Workflow

1. Modify Code Unit in library.
2. Generate patch with source map alignment.
3. Apply patch to all matching occurrences.
4. Validate with tests.
5. Commit change set (git).

## Extensibility

- `LanguageAdapter` interface for new languages.
- `Parser` provides AST nodes and spans.
- `Normalizer` maps nodes to standard Code Units.

## Safety & Idempotency

- Every destructive run requires explicit confirmation.
- Each ingestion creates a manifest with hashes.
- Re-ingestion of unchanged files is a no-op.

