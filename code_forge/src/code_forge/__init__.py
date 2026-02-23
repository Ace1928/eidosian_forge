"""Code Forge - static analysis, indexing, and reusable code library tooling."""

from code_forge.analyzer.code_indexer import (
    CodeElement,
    CodeIndexer,
    index_forge_codebase,
)
from code_forge.analyzer.generic_analyzer import GenericCodeAnalyzer
from code_forge.analyzer.python_analyzer import CodeAnalyzer
from code_forge.bench.runner import (
    BenchmarkConfig,
    BenchmarkResult,
    run_benchmark_suite,
)
from code_forge.canonicalize.planner import build_canonical_migration_plan
from code_forge.digester.drift import build_drift_report_from_output
from code_forge.digester.pipeline import (
    build_archive_reduction_plan,
    build_dependency_graph,
    build_duplication_index,
    build_repo_index,
    build_triage_report,
    run_archive_digester,
)
from code_forge.digester.schema import validate_output_dir
from code_forge.eval_os import (
    ArtifactContract,
    EvalConfig,
    EvalConfigMatrix,
    EvalRunOptions,
    FreshnessRecord,
    TaskSpec,
    compute_staleness_metrics,
    create_sample_config_matrix,
    create_sample_taskbank,
    load_eval_config_matrix,
    load_freshness_records,
    load_taskbank,
    run_eval_suite,
    write_eval_config_matrix,
    write_taskbank,
)
from code_forge.ingest.runner import IngestionRunner, IngestionStats
from code_forge.integration.memory import sync_units_to_memory_forge
from code_forge.integration.pipeline import (
    export_units_for_graphrag,
    sync_units_to_knowledge_forge,
)
from code_forge.integration.provenance import (
    read_provenance_links,
    write_provenance_links,
)
from code_forge.integration.provenance_registry import (
    build_provenance_registry,
    load_latest_benchmark_for_root,
    read_provenance_registry,
    write_provenance_registry,
)
from code_forge.librarian.core import CodeLibrarian
from code_forge.library.db import CodeLibraryDB, CodeUnit
from code_forge.library.similarity import (
    build_fingerprint,
    hamming_distance64,
    normalize_code_text,
    normalized_hash,
    simhash64,
    structural_hash,
    structural_normalize_code_text,
    token_jaccard,
    tokenize_code_text,
)
from code_forge.reconstruct.pipeline import (
    apply_reconstruction,
    build_reconstruction_from_library,
    compare_tree_parity,
    run_roundtrip_pipeline,
)
from code_forge.reconstruct.schema import validate_roundtrip_workspace

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
    "build_archive_reduction_plan",
    "build_drift_report_from_output",
    "build_duplication_index",
    "build_repo_index",
    "build_triage_report",
    "run_archive_digester",
    "validate_output_dir",
    "ArtifactContract",
    "EvalConfig",
    "EvalConfigMatrix",
    "EvalRunOptions",
    "FreshnessRecord",
    "TaskSpec",
    "compute_staleness_metrics",
    "create_sample_config_matrix",
    "create_sample_taskbank",
    "load_eval_config_matrix",
    "load_freshness_records",
    "load_taskbank",
    "run_eval_suite",
    "write_eval_config_matrix",
    "write_taskbank",
    "IngestionRunner",
    "IngestionStats",
    "export_units_for_graphrag",
    "sync_units_to_memory_forge",
    "sync_units_to_knowledge_forge",
    "write_provenance_links",
    "read_provenance_links",
    "build_provenance_registry",
    "write_provenance_registry",
    "read_provenance_registry",
    "load_latest_benchmark_for_root",
    "build_fingerprint",
    "hamming_distance64",
    "normalize_code_text",
    "normalized_hash",
    "simhash64",
    "structural_hash",
    "structural_normalize_code_text",
    "token_jaccard",
    "tokenize_code_text",
    "build_reconstruction_from_library",
    "compare_tree_parity",
    "apply_reconstruction",
    "run_roundtrip_pipeline",
    "validate_roundtrip_workspace",
]
