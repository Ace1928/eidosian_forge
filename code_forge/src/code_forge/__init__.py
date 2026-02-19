"""Code Forge - static analysis, indexing, and reusable code library tooling."""

from code_forge.analyzer.code_indexer import CodeElement, CodeIndexer, index_forge_codebase
from code_forge.analyzer.generic_analyzer import GenericCodeAnalyzer
from code_forge.analyzer.python_analyzer import CodeAnalyzer
from code_forge.bench.runner import BenchmarkConfig, BenchmarkResult, run_benchmark_suite
from code_forge.canonicalize.planner import build_canonical_migration_plan
from code_forge.digester.pipeline import (
    build_dependency_graph,
    build_duplication_index,
    build_repo_index,
    build_triage_report,
    run_archive_digester,
)
from code_forge.ingest.runner import IngestionRunner, IngestionStats
from code_forge.integration.pipeline import export_units_for_graphrag, sync_units_to_knowledge_forge
from code_forge.librarian.core import CodeLibrarian
from code_forge.library.db import CodeLibraryDB, CodeUnit
from code_forge.library.similarity import (
    build_fingerprint,
    hamming_distance64,
    normalize_code_text,
    normalized_hash,
    simhash64,
    token_jaccard,
    tokenize_code_text,
)

__all__ = [
    "CodeAnalyzer",
    "GenericCodeAnalyzer",
    "CodeIndexer",
    "CodeElement",
    "BenchmarkConfig",
    "BenchmarkResult",
    "index_forge_codebase",
    "run_benchmark_suite",
    "build_canonical_migration_plan",
    "CodeLibrarian",
    "CodeLibraryDB",
    "CodeUnit",
    "build_dependency_graph",
    "build_duplication_index",
    "build_repo_index",
    "build_triage_report",
    "run_archive_digester",
    "IngestionRunner",
    "IngestionStats",
    "export_units_for_graphrag",
    "sync_units_to_knowledge_forge",
    "build_fingerprint",
    "hamming_distance64",
    "normalize_code_text",
    "normalized_hash",
    "simhash64",
    "token_jaccard",
    "tokenize_code_text",
]
