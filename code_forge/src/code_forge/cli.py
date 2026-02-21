#!/usr/bin/env python3
"""
Code Forge CLI - Command-line interface for code analysis and indexing.

Standalone Usage:
    code-forge status              # Show code index status
    code-forge analyze <file>      # Analyze a Python file
    code-forge index <path>        # Index codebase
    code-forge search <query>      # Search code elements
    code-forge ingest <file>       # Ingest into code library

Enhanced with other forges:
    - knowledge_forge: Semantic code understanding
    - llm_forge: AI-powered code analysis
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

from eidosian_core import eidosian

# Add lib to path for CLI framework
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "lib"))

from cli import CommandResult, ForgeDetector, StandardCLI

from code_forge import (
    CodeAnalyzer,
    CodeIndexer,
    CodeLibrarian,
    CodeLibraryDB,
    GenericCodeAnalyzer,
    IngestionRunner,
    apply_reconstruction,
    build_canonical_migration_plan,
    build_dependency_graph,
    build_drift_report_from_output,
    build_duplication_index,
    build_reconstruction_from_library,
    build_repo_index,
    build_triage_report,
    compare_tree_parity,
    export_units_for_graphrag,
    run_archive_digester,
    run_benchmark_suite,
    run_roundtrip_pipeline,
    sync_units_to_knowledge_forge,
    compute_staleness_metrics,
    create_sample_config_matrix,
    create_sample_taskbank,
    load_freshness_records,
    run_eval_suite,
    EvalRunOptions,
    validate_output_dir,
    validate_roundtrip_workspace,
)

# Default paths
FORGE_ROOT = Path(
    os.environ.get("EIDOS_FORGE_DIR", str(Path(__file__).resolve().parents[3]))
).resolve()
DEFAULT_INDEX_PATH = FORGE_ROOT / "data" / "code_index.json"
DEFAULT_LIBRARY_PATH = FORGE_ROOT / "data" / "code_library.json"
DEFAULT_DB_PATH = FORGE_ROOT / "data" / "code_forge" / "library.sqlite"
DEFAULT_RUNS_DIR = FORGE_ROOT / "data" / "code_forge" / "ingestion_runs"


class CodeForgeCLI(StandardCLI):
    """CLI for Code Forge - code analysis and indexing."""

    name = "code_forge"
    description = "AST-based code analysis, indexing, and library management"
    version = "1.0.0"

    def __init__(self):
        super().__init__()
        self._analyzer: Optional[CodeAnalyzer] = None
        self._indexer: Optional[CodeIndexer] = None
        self._librarian: Optional[CodeLibrarian] = None
        self._library_db: Optional[CodeLibraryDB] = None
        self._runner: Optional[IngestionRunner] = None

    @property
    def analyzer(self) -> CodeAnalyzer:
        """Lazy-load code analyzer."""
        if self._analyzer is None:
            self._analyzer = CodeAnalyzer()
        return self._analyzer

    @property
    def indexer(self) -> CodeIndexer:
        """Lazy-load code indexer."""
        if self._indexer is None:
            self._indexer = CodeIndexer(DEFAULT_INDEX_PATH)
        return self._indexer

    @property
    def librarian(self) -> CodeLibrarian:
        """Lazy-load code librarian."""
        if self._librarian is None:
            self._librarian = CodeLibrarian(DEFAULT_LIBRARY_PATH)
        return self._librarian

    @property
    def library_db(self) -> CodeLibraryDB:
        if self._library_db is None:
            self._library_db = CodeLibraryDB(DEFAULT_DB_PATH)
        return self._library_db

    @property
    def runner(self) -> IngestionRunner:
        if self._runner is None:
            self._runner = IngestionRunner(
                db=self.library_db, runs_dir=DEFAULT_RUNS_DIR
            )
        return self._runner

    @eidosian()
    def register_commands(self, subparsers) -> None:
        """Register code-forge specific commands."""

        # Analyze command
        analyze_parser = subparsers.add_parser(
            "analyze",
            help="Analyze a Python file",
        )
        analyze_parser.add_argument(
            "path",
            help="Path to Python file",
        )
        analyze_parser.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="Show detailed analysis",
        )
        analyze_parser.set_defaults(func=self._cmd_analyze)

        # Index command
        index_parser = subparsers.add_parser(
            "index",
            help="Index a codebase",
        )
        index_parser.add_argument(
            "path",
            nargs="?",
            default=str(FORGE_ROOT),
            help="Path to index (default: eidosian_forge)",
        )
        index_parser.add_argument(
            "--full",
            action="store_true",
            help="Force full reindex",
        )
        index_parser.set_defaults(func=self._cmd_index)

        # Search command
        search_parser = subparsers.add_parser(
            "search",
            help="Search code elements",
        )
        search_parser.add_argument(
            "query",
            help="Search query",
        )
        search_parser.add_argument(
            "-t",
            "--type",
            choices=["function", "class", "method", "module"],
            help="Filter by element type",
        )
        search_parser.add_argument(
            "-n",
            "--limit",
            type=int,
            default=10,
            help="Maximum results (default: 10)",
        )
        search_parser.set_defaults(func=self._cmd_search)

        # Ingest command
        ingest_parser = subparsers.add_parser(
            "ingest",
            help="Ingest file into code library",
        )
        ingest_parser.add_argument(
            "path",
            help="Path to file",
        )
        ingest_parser.set_defaults(func=self._cmd_ingest)

        ingest_dir_parser = subparsers.add_parser(
            "ingest-dir",
            help="Ingest a directory into the code library (non-destructive)",
        )
        ingest_dir_parser.add_argument(
            "path",
            help="Path to directory",
        )
        ingest_dir_parser.add_argument(
            "--mode",
            choices=["analysis", "archival"],
            default="analysis",
            help="Ingestion mode (default: analysis)",
        )
        ingest_dir_parser.add_argument(
            "--ext",
            nargs="*",
            default=None,
            help="File extensions to include (default: multi-language set)",
        )
        ingest_dir_parser.add_argument(
            "--max-files",
            type=int,
            default=None,
            help="Maximum number of files to ingest (default: unlimited)",
        )
        ingest_dir_parser.add_argument(
            "--progress-every",
            type=int,
            default=50,
            help="Write progress manifest every N files (default: 50)",
        )
        ingest_dir_parser.set_defaults(func=self._cmd_ingest_dir)

        ingest_bg_parser = subparsers.add_parser(
            "ingest-bg",
            help="Start background ingestion process (non-blocking)",
        )
        ingest_bg_parser.add_argument("path", help="Path to directory")
        ingest_bg_parser.add_argument(
            "--mode",
            choices=["analysis", "archival"],
            default="analysis",
            help="Ingestion mode (default: analysis)",
        )
        ingest_bg_parser.add_argument(
            "--ext",
            nargs="*",
            default=None,
            help="File extensions to include (default: multi-language set)",
        )
        ingest_bg_parser.add_argument(
            "--max-files",
            type=int,
            default=None,
            help="Maximum number of files to ingest (default: unlimited)",
        )
        ingest_bg_parser.add_argument(
            "--progress-every",
            type=int,
            default=100,
            help="Write progress manifest every N files (default: 100)",
        )
        ingest_bg_parser.set_defaults(func=self._cmd_ingest_bg)

        ingest_status_parser = subparsers.add_parser(
            "ingest-status",
            help="Check ingestion progress by run_id",
        )
        ingest_status_parser.add_argument("run_id", help="Run ID to inspect")
        ingest_status_parser.set_defaults(func=self._cmd_ingest_status)

        # Library command
        lib_parser = subparsers.add_parser(
            "library",
            help="List code library contents",
        )
        lib_parser.add_argument(
            "-n",
            "--limit",
            type=int,
            default=20,
            help="Maximum items (default: 20)",
        )
        lib_parser.set_defaults(func=self._cmd_library)

        # Stats command
        stats_parser = subparsers.add_parser(
            "stats",
            help="Show detailed statistics",
        )
        stats_parser.set_defaults(func=self._cmd_stats)

        dedup_parser = subparsers.add_parser(
            "dedup-report",
            help="Report duplicate code units by content hash",
        )
        dedup_parser.add_argument(
            "--min-occurrences",
            type=int,
            default=2,
            help="Minimum duplicate count to include (default: 2)",
        )
        dedup_parser.add_argument(
            "--limit-groups",
            type=int,
            default=100,
            help="Maximum duplicate groups (default: 100)",
        )
        dedup_parser.set_defaults(func=self._cmd_dedup_report)

        norm_dedup_parser = subparsers.add_parser(
            "normalized-dedup-report",
            help="Report duplicate code units by normalized fingerprints",
        )
        norm_dedup_parser.add_argument(
            "--min-occurrences",
            type=int,
            default=2,
            help="Minimum duplicate count to include (default: 2)",
        )
        norm_dedup_parser.add_argument(
            "--limit-groups",
            type=int,
            default=100,
            help="Maximum duplicate groups (default: 100)",
        )
        norm_dedup_parser.set_defaults(func=self._cmd_normalized_dedup_report)

        near_dedup_parser = subparsers.add_parser(
            "near-dedup-report",
            help="Report near-duplicate code units using simhash distance",
        )
        near_dedup_parser.add_argument(
            "--max-hamming",
            type=int,
            default=6,
            help="Maximum hamming distance for near-duplicates (default: 6)",
        )
        near_dedup_parser.add_argument(
            "--min-tokens",
            type=int,
            default=20,
            help="Minimum token count for candidate units (default: 20)",
        )
        near_dedup_parser.add_argument(
            "--limit-pairs",
            type=int,
            default=200,
            help="Maximum near-duplicate pairs to return (default: 200)",
        )
        near_dedup_parser.add_argument(
            "--language",
            default=None,
            help="Optional language filter (e.g., python, javascript, go)",
        )
        near_dedup_parser.add_argument(
            "--type",
            default=None,
            help="Optional unit type filter (e.g., function, class, method)",
        )
        near_dedup_parser.set_defaults(func=self._cmd_near_dedup_report)

        semantic_parser = subparsers.add_parser(
            "semantic-search",
            help="Hybrid semantic/code search across indexed units",
        )
        semantic_parser.add_argument("query", help="Search query")
        semantic_parser.add_argument(
            "-n",
            "--limit",
            type=int,
            default=20,
            help="Maximum results (default: 20)",
        )
        semantic_parser.add_argument(
            "--language",
            default=None,
            help="Optional language filter",
        )
        semantic_parser.add_argument(
            "--type",
            default=None,
            help="Optional unit type filter",
        )
        semantic_parser.add_argument(
            "--min-score",
            type=float,
            default=0.05,
            help="Minimum semantic score threshold (default: 0.05)",
        )
        semantic_parser.set_defaults(func=self._cmd_semantic_search)

        trace_parser = subparsers.add_parser(
            "trace",
            help="Trace contains-graph from a unit (qualified name or unit id)",
        )
        trace_parser.add_argument(
            "target",
            help="Qualified name (e.g. pkg.module.Class) or unit id hash",
        )
        trace_parser.add_argument(
            "--depth",
            type=int,
            default=3,
            help="Traversal depth (default: 3)",
        )
        trace_parser.add_argument(
            "--max-nodes",
            type=int,
            default=300,
            help="Maximum nodes in trace output (default: 300)",
        )
        trace_parser.set_defaults(func=self._cmd_trace)

        sync_kb_parser = subparsers.add_parser(
            "sync-knowledge",
            help="Sync indexed code units into Knowledge Forge graph",
        )
        sync_kb_parser.add_argument(
            "--kb-path",
            default=str(FORGE_ROOT / "data" / "kb.json"),
            help="Knowledge graph persistence path (default: data/kb.json)",
        )
        sync_kb_parser.add_argument(
            "--limit",
            type=int,
            default=5000,
            help="Maximum units to sync (default: 5000)",
        )
        sync_kb_parser.add_argument(
            "--min-tokens",
            type=int,
            default=5,
            help="Minimum token count to include (default: 5)",
        )
        sync_kb_parser.add_argument(
            "--run-id",
            default=None,
            help="Optional ingestion run_id scope (default: latest/global query path)",
        )
        sync_kb_parser.set_defaults(func=self._cmd_sync_knowledge)

        export_grag_parser = subparsers.add_parser(
            "export-graphrag",
            help="Export code units as GraphRAG-ready text corpus",
        )
        export_grag_parser.add_argument(
            "--output-dir",
            default=str(FORGE_ROOT / "data" / "code_forge" / "graphrag_input"),
            help="Output directory for GraphRAG corpus documents",
        )
        export_grag_parser.add_argument(
            "--limit",
            type=int,
            default=20000,
            help="Maximum units to export (default: 20000)",
        )
        export_grag_parser.add_argument(
            "--min-tokens",
            type=int,
            default=5,
            help="Minimum token count to include (default: 5)",
        )
        export_grag_parser.add_argument(
            "--run-id",
            default=None,
            help="Optional ingestion run_id scope (default: latest/global query path)",
        )
        export_grag_parser.set_defaults(func=self._cmd_export_graphrag)

        catalog_parser = subparsers.add_parser(
            "catalog",
            help="Build deterministic repo + duplication intake artifacts",
        )
        catalog_parser.add_argument(
            "path",
            nargs="?",
            default=str(FORGE_ROOT),
            help="Root path to catalog (default: forge root)",
        )
        catalog_parser.add_argument(
            "--output-dir",
            default=str(FORGE_ROOT / "data" / "code_forge" / "digester" / "latest"),
            help="Output directory for intake artifacts",
        )
        catalog_parser.add_argument(
            "--ext",
            nargs="*",
            default=None,
            help="File extensions to include (default: multi-language set)",
        )
        catalog_parser.add_argument(
            "--max-files",
            type=int,
            default=None,
            help="Maximum files to catalog (default: unlimited)",
        )
        catalog_parser.set_defaults(func=self._cmd_catalog)

        triage_parser = subparsers.add_parser(
            "triage-report",
            help="Generate explainable keep/extract/refactor/quarantine/delete triage reports",
        )
        triage_parser.add_argument(
            "--output-dir",
            default=str(FORGE_ROOT / "data" / "code_forge" / "digester" / "latest"),
            help="Directory containing repo_index.json/duplication_index.json and triage outputs",
        )
        triage_parser.set_defaults(func=self._cmd_triage_report)

        digest_parser = subparsers.add_parser(
            "digest",
            help="Run end-to-end archive digester (ingest -> catalog -> dedup -> triage -> optional exports)",
        )
        digest_parser.add_argument(
            "path",
            nargs="?",
            default=str(FORGE_ROOT),
            help="Root path to process (default: forge root)",
        )
        digest_parser.add_argument(
            "--mode",
            choices=["analysis", "archival"],
            default="analysis",
            help="Ingestion mode (default: analysis)",
        )
        digest_parser.add_argument(
            "--output-dir",
            default=str(FORGE_ROOT / "data" / "code_forge" / "digester" / "latest"),
            help="Output directory for digestion artifacts",
        )
        digest_parser.add_argument(
            "--ext",
            nargs="*",
            default=None,
            help="File extensions to include (default: multi-language set)",
        )
        digest_parser.add_argument(
            "--max-files",
            type=int,
            default=None,
            help="Maximum files to process (default: unlimited)",
        )
        digest_parser.add_argument(
            "--progress-every",
            type=int,
            default=200,
            help="Write ingestion progress every N files (default: 200)",
        )
        digest_parser.add_argument(
            "--sync-knowledge",
            action="store_true",
            help="Sync digested code units into Knowledge Forge",
        )
        digest_parser.add_argument(
            "--sync-memory",
            action="store_true",
            help="Sync digested code units into Memory Forge JSON episodic store",
        )
        digest_parser.add_argument(
            "--kb-path",
            default=str(FORGE_ROOT / "data" / "kb.json"),
            help="Knowledge graph path when --sync-knowledge is enabled",
        )
        digest_parser.add_argument(
            "--memory-path",
            default=str(FORGE_ROOT / "data" / "episodic_memory.json"),
            help="Memory store path when --sync-memory is enabled",
        )
        digest_parser.add_argument(
            "--export-graphrag",
            action="store_true",
            help="Export GraphRAG-ready text corpus after digestion",
        )
        digest_parser.add_argument(
            "--graphrag-output-dir",
            default=str(FORGE_ROOT / "data" / "code_forge" / "graphrag_input"),
            help="GraphRAG export output directory",
        )
        digest_parser.add_argument(
            "--graph-export-limit",
            type=int,
            default=20000,
            help="Maximum units for knowledge/GraphRAG export (default: 20000)",
        )
        digest_parser.add_argument(
            "--integration-policy",
            choices=["run", "effective_run", "global"],
            default="effective_run",
            help="Scope policy for integration exports",
        )
        digest_parser.add_argument(
            "--no-strict-validation",
            action="store_true",
            help="Disable strict artifact schema validation failure mode",
        )
        digest_parser.add_argument(
            "--no-drift-report",
            action="store_true",
            help="Disable drift report generation",
        )
        digest_parser.add_argument(
            "--no-history-snapshot",
            action="store_true",
            help="Disable drift history snapshot writes",
        )
        digest_parser.add_argument(
            "--previous-snapshot",
            default=None,
            help="Optional path to previous drift history snapshot JSON",
        )
        digest_parser.add_argument(
            "--drift-warn-pct",
            type=float,
            default=30.0,
            help="Warning threshold for relative drift delta percent",
        )
        digest_parser.add_argument(
            "--drift-min-abs-delta",
            type=float,
            default=1.0,
            help="Warning threshold for absolute drift delta",
        )
        digest_parser.set_defaults(func=self._cmd_digest)

        dep_graph_parser = subparsers.add_parser(
            "dependency-graph",
            help="Build code dependency graph from imports/calls/uses relationships",
        )
        dep_graph_parser.add_argument(
            "--output-dir",
            default=str(FORGE_ROOT / "data" / "code_forge" / "digester" / "latest"),
            help="Output directory for dependency_graph.json",
        )
        dep_graph_parser.add_argument(
            "--rel-types",
            nargs="*",
            default=["imports", "calls", "uses"],
            help="Relationship types to include",
        )
        dep_graph_parser.add_argument(
            "--limit-edges",
            type=int,
            default=20000,
            help="Maximum relationship rows to aggregate",
        )
        dep_graph_parser.set_defaults(func=self._cmd_dependency_graph)

        validate_parser = subparsers.add_parser(
            "validate-artifacts",
            help="Validate digester artifacts against strict schema contracts",
        )
        validate_parser.add_argument(
            "--output-dir",
            default=str(FORGE_ROOT / "data" / "code_forge" / "digester" / "latest"),
            help="Digester artifact directory to validate",
        )
        validate_parser.set_defaults(func=self._cmd_validate_artifacts)

        validate_roundtrip_parser = subparsers.add_parser(
            "validate-roundtrip",
            help="Validate roundtrip reconstruction/parity/apply artifacts in a workspace",
        )
        validate_roundtrip_parser.add_argument(
            "--workspace-dir",
            required=True,
            help="Roundtrip workspace directory containing summary/manifest/parity artifacts",
        )
        validate_roundtrip_parser.add_argument(
            "--require-apply-report",
            action="store_true",
            help="Fail if apply report is missing from transaction backup directory",
        )
        validate_roundtrip_parser.add_argument(
            "--verify-hashes",
            action="store_true",
            help="Verify reconstructed file hashes against reconstruction manifest",
        )
        validate_roundtrip_parser.set_defaults(func=self._cmd_validate_roundtrip)

        drift_parser = subparsers.add_parser(
            "drift-report",
            help="Generate/update drift report from digester artifacts",
        )
        drift_parser.add_argument(
            "--output-dir",
            default=str(FORGE_ROOT / "data" / "code_forge" / "digester" / "latest"),
            help="Digester artifact directory",
        )
        drift_parser.add_argument(
            "--previous-snapshot",
            default=None,
            help="Optional path to previous drift history snapshot JSON",
        )
        drift_parser.add_argument(
            "--no-history-snapshot",
            action="store_true",
            help="Disable drift history snapshot writes",
        )
        drift_parser.add_argument(
            "--drift-warn-pct",
            type=float,
            default=30.0,
            help="Warning threshold for relative drift delta percent",
        )
        drift_parser.add_argument(
            "--drift-min-abs-delta",
            type=float,
            default=1.0,
            help="Warning threshold for absolute drift delta",
        )
        drift_parser.set_defaults(func=self._cmd_drift_report)

        benchmark_parser = subparsers.add_parser(
            "benchmark",
            help="Run ingestion/search/dependency benchmark suite with optional regression gates",
        )
        benchmark_parser.add_argument(
            "--root",
            default=str(FORGE_ROOT),
            help="Root path for benchmark ingestion workload",
        )
        benchmark_parser.add_argument(
            "--output",
            default=str(FORGE_ROOT / "reports" / "code_forge_benchmark_latest.json"),
            help="Output JSON report path",
        )
        benchmark_parser.add_argument(
            "--ext",
            nargs="*",
            default=None,
            help="Extensions to include (default: multi-language set)",
        )
        benchmark_parser.add_argument(
            "--max-files",
            type=int,
            default=2000,
            help="Max files for benchmark ingestion",
        )
        benchmark_parser.add_argument(
            "--ingestion-repeats",
            type=int,
            default=1,
            help="Ingestion benchmark repeat count",
        )
        benchmark_parser.add_argument(
            "--query-repeats",
            type=int,
            default=5,
            help="Per-query repeat count for semantic search benchmark",
        )
        benchmark_parser.add_argument(
            "--query",
            action="append",
            default=[],
            help="Add benchmark query (repeat option for multiple queries)",
        )
        benchmark_parser.add_argument(
            "--baseline",
            default=str(FORGE_ROOT / "reports" / "code_forge_benchmark_baseline.json"),
            help="Baseline JSON path for regression gates",
        )
        benchmark_parser.add_argument(
            "--max-regression-pct",
            type=float,
            default=25.0,
            help="Allowed regression percent before gate fails",
        )
        benchmark_parser.add_argument(
            "--skip-prepare-ingest",
            action="store_true",
            help="Skip ingestion prep on primary DB before search/graph benchmarks",
        )
        benchmark_parser.add_argument(
            "--write-baseline",
            action="store_true",
            help="Write current run as new baseline at --baseline path",
        )
        benchmark_parser.set_defaults(func=self._cmd_benchmark)

        eval_init_parser = subparsers.add_parser(
            "eval-init",
            help="Create sample TaskBank and config-matrix files for eval/observability runs",
        )
        eval_init_parser.add_argument(
            "--taskbank",
            default=str(FORGE_ROOT / "config" / "eval" / "taskbank.json"),
            help="Output path for taskbank JSON",
        )
        eval_init_parser.add_argument(
            "--matrix",
            default=str(FORGE_ROOT / "config" / "eval" / "config_matrix.json"),
            help="Output path for config matrix JSON",
        )
        eval_init_parser.add_argument(
            "--force",
            action="store_true",
            help="Overwrite existing files if present",
        )
        eval_init_parser.set_defaults(func=self._cmd_eval_init)

        eval_run_parser = subparsers.add_parser(
            "eval-run",
            help="Run taskbank x config-matrix evaluation with full trace/replay artifacts",
        )
        eval_run_parser.add_argument(
            "--taskbank",
            default=str(FORGE_ROOT / "config" / "eval" / "taskbank.json"),
            help="TaskBank JSON path",
        )
        eval_run_parser.add_argument(
            "--matrix",
            default=str(FORGE_ROOT / "config" / "eval" / "config_matrix.json"),
            help="Config-matrix JSON path",
        )
        eval_run_parser.add_argument(
            "--output-dir",
            default=str(FORGE_ROOT / "reports" / "code_forge_eval"),
            help="Output directory for run traces and summary.json",
        )
        eval_run_parser.add_argument(
            "--repo-root",
            default=str(FORGE_ROOT),
            help="Repository root used for snapshot metadata and task workdir resolution",
        )
        eval_run_parser.add_argument(
            "--repeats",
            type=int,
            default=1,
            help="Repeat count per (task, config) pair",
        )
        eval_run_parser.add_argument(
            "--max-parallel",
            type=int,
            default=1,
            help="Maximum parallel runs",
        )
        eval_run_parser.add_argument(
            "--replay-mode",
            choices=["off", "record", "replay"],
            default="off",
            help="Replay behavior: off, record live outputs, or replay only",
        )
        eval_run_parser.add_argument(
            "--default-timeout-sec",
            type=int,
            default=1200,
            help="Fallback timeout when task timeout is not specified",
        )
        eval_run_parser.add_argument(
            "--staleness-log",
            default=None,
            help="Optional freshness/staleness log (json or jsonl) to include in summary",
        )
        eval_run_parser.set_defaults(func=self._cmd_eval_run)

        eval_staleness_parser = subparsers.add_parser(
            "eval-staleness",
            help="Compute staleness metrics from a freshness log payload",
        )
        eval_staleness_parser.add_argument(
            "--input",
            required=True,
            help="Freshness log path (.json or .jsonl)",
        )
        eval_staleness_parser.add_argument(
            "--output",
            default=None,
            help="Optional output JSON path for computed metrics",
        )
        eval_staleness_parser.set_defaults(func=self._cmd_eval_staleness)

        canonical_parser = subparsers.add_parser(
            "canonical-plan",
            help="Build canonical migration map and compatibility shim staging from triage artifacts",
        )
        canonical_parser.add_argument(
            "--triage-path",
            default=str(
                FORGE_ROOT
                / "data"
                / "code_forge"
                / "digester"
                / "latest"
                / "triage.json"
            ),
            help="Path to triage.json artifact",
        )
        canonical_parser.add_argument(
            "--output-dir",
            default=str(
                FORGE_ROOT / "data" / "code_forge" / "canonicalization" / "latest"
            ),
            help="Output directory for migration map and plan artifacts",
        )
        canonical_parser.add_argument(
            "--labels",
            nargs="*",
            default=["extract", "refactor", "delete_candidate"],
            help="Triage labels to include in canonical planning",
        )
        canonical_parser.add_argument(
            "--max-entries",
            type=int,
            default=500,
            help="Maximum triage entries to include",
        )
        canonical_parser.add_argument(
            "--no-shims",
            action="store_true",
            help="Disable compatibility shim generation",
        )
        canonical_parser.set_defaults(func=self._cmd_canonical_plan)

        reconstruct_parser = subparsers.add_parser(
            "reconstruct-project",
            help="Rebuild source files from Code Forge library file_records/content blobs",
        )
        reconstruct_parser.add_argument(
            "--source-root",
            required=True,
            help="Source root that was ingested (prefix filter for file_records)",
        )
        reconstruct_parser.add_argument(
            "--output-dir",
            required=True,
            help="Output directory for reconstructed files + manifest",
        )
        reconstruct_parser.add_argument(
            "--ext",
            nargs="*",
            default=None,
            help="Optional file extensions filter (default: all tracked files under source-root)",
        )
        reconstruct_parser.add_argument(
            "--max-files",
            type=int,
            default=None,
            help="Optional max file records to process",
        )
        reconstruct_parser.add_argument(
            "--no-strict",
            action="store_true",
            help="Allow missing content blobs without failing command",
        )
        reconstruct_parser.set_defaults(func=self._cmd_reconstruct_project)

        parity_parser = subparsers.add_parser(
            "parity-report",
            help="Compare source tree and reconstructed tree content hashes",
        )
        parity_parser.add_argument(
            "--source-root",
            required=True,
            help="Original source root path",
        )
        parity_parser.add_argument(
            "--reconstructed-root",
            required=True,
            help="Reconstructed tree root path",
        )
        parity_parser.add_argument(
            "--report-path",
            default=None,
            help="Optional output path for parity_report.json",
        )
        parity_parser.add_argument(
            "--ext",
            nargs="*",
            default=None,
            help="Optional extension filter for parity checks",
        )
        parity_parser.set_defaults(func=self._cmd_parity_report)

        apply_parser = subparsers.add_parser(
            "apply-reconstruction",
            help="Apply reconstructed tree onto target root with transactional backup",
        )
        apply_parser.add_argument(
            "--reconstructed-root",
            required=True,
            help="Reconstructed tree root path",
        )
        apply_parser.add_argument(
            "--target-root",
            required=True,
            help="Target root to replace/sync",
        )
        apply_parser.add_argument(
            "--backup-root",
            default=str(FORGE_ROOT / "Backups" / "code_forge_roundtrip"),
            help="Backup directory for transactional apply artifacts",
        )
        apply_parser.add_argument(
            "--parity-report",
            default=None,
            help="Optional parity report JSON path to enforce before apply",
        )
        apply_parser.add_argument(
            "--allow-non-parity",
            action="store_true",
            help="Allow apply even if parity report fails",
        )
        apply_parser.add_argument(
            "--no-prune",
            action="store_true",
            help="Do not remove files missing from reconstructed tree",
        )
        apply_parser.add_argument(
            "--require-manifest",
            action="store_true",
            help="Require reconstruction manifest-managed paths for apply safety",
        )
        apply_parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Compute apply plan without modifying files",
        )
        apply_parser.set_defaults(func=self._cmd_apply_reconstruction)

        roundtrip_parser = subparsers.add_parser(
            "roundtrip",
            help="Run digest + integration exports + reconstruction + parity (+optional apply)",
        )
        roundtrip_parser.add_argument(
            "path",
            nargs="?",
            default=str(FORGE_ROOT),
            help="Root path to process",
        )
        roundtrip_parser.add_argument(
            "--workspace-dir",
            required=True,
            help="Workspace directory for digester/reconstruction/parity artifacts",
        )
        roundtrip_parser.add_argument(
            "--mode",
            choices=["analysis", "archival"],
            default="analysis",
            help="Ingestion mode (default: analysis)",
        )
        roundtrip_parser.add_argument(
            "--ext",
            nargs="*",
            default=None,
            help="Optional file extensions include-list",
        )
        roundtrip_parser.add_argument(
            "--max-files",
            type=int,
            default=None,
            help="Optional max files to process",
        )
        roundtrip_parser.add_argument(
            "--progress-every",
            type=int,
            default=200,
            help="Write ingestion progress every N files",
        )
        roundtrip_parser.add_argument(
            "--sync-knowledge",
            action="store_true",
            help="Sync digested code units into Knowledge Forge",
        )
        roundtrip_parser.add_argument(
            "--sync-memory",
            action="store_true",
            help="Sync digested code units into Memory Forge JSON episodic store",
        )
        roundtrip_parser.add_argument(
            "--kb-path",
            default=str(FORGE_ROOT / "data" / "kb.json"),
            help="Knowledge graph persistence path",
        )
        roundtrip_parser.add_argument(
            "--memory-path",
            default=str(FORGE_ROOT / "data" / "episodic_memory.json"),
            help="Memory store path when --sync-memory is enabled",
        )
        roundtrip_parser.add_argument(
            "--export-graphrag",
            action="store_true",
            help="Export GraphRAG-ready corpus from indexed units",
        )
        roundtrip_parser.add_argument(
            "--graphrag-output-dir",
            default=str(FORGE_ROOT / "data" / "code_forge" / "graphrag_input"),
            help="GraphRAG export output directory",
        )
        roundtrip_parser.add_argument(
            "--graph-export-limit",
            type=int,
            default=20000,
            help="Maximum units for Knowledge/GraphRAG exports",
        )
        roundtrip_parser.add_argument(
            "--integration-policy",
            choices=["run", "effective_run", "global"],
            default="effective_run",
            help="Scope policy for integration exports",
        )
        roundtrip_parser.add_argument(
            "--apply",
            action="store_true",
            help="Apply reconstructed tree to target after parity pass",
        )
        roundtrip_parser.add_argument(
            "--backup-root",
            default=str(FORGE_ROOT / "Backups" / "code_forge_roundtrip"),
            help="Backup directory used when --apply is enabled",
        )
        roundtrip_parser.add_argument(
            "--no-strict-validation",
            action="store_true",
            help="Disable strict schema/blob validation failure mode",
        )
        roundtrip_parser.set_defaults(func=self._cmd_roundtrip)

    @eidosian()
    def cmd_status(self, args) -> CommandResult:
        """Show code forge status."""
        try:
            # Check index
            index_exists = DEFAULT_INDEX_PATH.exists()
            element_count = len(self.indexer.elements) if index_exists else 0

            # Count by type
            type_counts = {}
            for elem in self.indexer.elements.values():
                t = elem.element_type
                type_counts[t] = type_counts.get(t, 0) + 1

            # Check library
            lib_exists = DEFAULT_LIBRARY_PATH.exists()

            integrations = []
            if ForgeDetector.is_available("knowledge_forge"):
                integrations.append("knowledge_forge")
            if ForgeDetector.is_available("llm_forge"):
                integrations.append("llm_forge")
            if ForgeDetector.is_available("graphrag"):
                integrations.append("graphrag")

            db_total = self.library_db.count_units()
            db_by_type = self.library_db.count_units_by_type()
            db_by_language = self.library_db.count_units_by_language()
            rel_counts = self.library_db.relationship_counts()
            digester_dir = FORGE_ROOT / "data" / "code_forge" / "digester" / "latest"
            latest_runs = self.library_db.latest_runs(limit=5)

            return CommandResult(
                True,
                f"Code Forge operational - {db_total} units in library",
                {
                    "elements_indexed": element_count,
                    "types": type_counts,
                    "index_path": str(DEFAULT_INDEX_PATH),
                    "index_exists": index_exists,
                    "library_exists": lib_exists,
                    "db_total_units": db_total,
                    "db_units_by_type": db_by_type,
                    "db_units_by_language": db_by_language,
                    "db_relationship_counts": rel_counts,
                    "latest_ingestion_runs": latest_runs,
                    "supported_extensions": sorted(
                        GenericCodeAnalyzer.supported_extensions()
                    ),
                    "digester_artifacts": {
                        "output_dir": str(digester_dir),
                        "repo_index": str(digester_dir / "repo_index.json"),
                        "duplication_index": str(
                            digester_dir / "duplication_index.json"
                        ),
                        "dependency_graph": str(digester_dir / "dependency_graph.json"),
                        "triage": str(digester_dir / "triage.json"),
                        "triage_audit": str(digester_dir / "triage_audit.json"),
                        "drift_report": str(digester_dir / "drift_report.json"),
                        "drift_history_dir": str(digester_dir / "history"),
                    },
                    "integrations": integrations,
                },
            )
        except Exception as e:
            return CommandResult(False, f"Error: {e}")

    def _cmd_analyze(self, args) -> None:
        """Analyze a Python file."""
        try:
            path = Path(args.path)
            if not path.exists():
                result = CommandResult(False, f"File not found: {path}")
            elif not path.is_file():
                result = CommandResult(False, "Path must be a file")
            else:
                analysis = self.analyzer.analyze_file(path)

                if args.verbose:
                    result = CommandResult(True, f"Analyzed {path.name}", analysis)
                else:
                    summary = {
                        "file": str(path),
                        "functions": len(analysis.get("functions", [])),
                        "classes": len(analysis.get("classes", [])),
                        "imports": len(analysis.get("imports", [])),
                    }
                    result = CommandResult(
                        True,
                        f"{path.name}: {summary['functions']} functions, {summary['classes']} classes",
                        summary,
                    )
        except Exception as e:
            result = CommandResult(False, f"Analysis error: {e}")
        self._output(result, args)

    def _cmd_index(self, args) -> None:
        """Index a codebase."""
        try:
            path = Path(args.path)
            if not path.exists():
                result = CommandResult(False, f"Path not found: {path}")
            else:
                count = self.indexer.index_directory(path)
                result = CommandResult(
                    True,
                    f"Indexed {count} elements from {path}",
                    {
                        "path": str(path),
                        "elements_indexed": count,
                        "total_elements": len(self.indexer.elements),
                    },
                )
        except Exception as e:
            result = CommandResult(False, f"Indexing error: {e}")
        self._output(result, args)

    def _cmd_search(self, args) -> None:
        """Search code elements."""
        try:
            results = []
            query_lower = args.query.lower()

            for elem in self.indexer.elements.values():
                # Filter by type if specified
                if args.type and elem.element_type != args.type:
                    continue

                # Match by name or docstring
                if (
                    query_lower in elem.name.lower()
                    or query_lower in elem.qualified_name.lower()
                    or (elem.docstring and query_lower in elem.docstring.lower())
                ):
                    results.append(
                        {
                            "name": elem.name,
                            "type": elem.element_type,
                            "qualified_name": elem.qualified_name,
                            "file": elem.file_path,
                            "line": elem.line_start,
                        }
                    )
                    if len(results) >= args.limit:
                        break

            result = CommandResult(
                True,
                f"Found {len(results)} matches for '{args.query}'",
                {"results": results, "query": args.query},
            )
        except Exception as e:
            result = CommandResult(False, f"Search error: {e}")
        self._output(result, args)

    def _cmd_ingest(self, args) -> None:
        """Ingest file into code library."""
        try:
            path = Path(args.path)
            if not path.is_file():
                result = CommandResult(False, "Path must be a file")
            else:
                analysis = self.analyzer.analyze_file(path)
                content = path.read_text(encoding="utf-8")

                sid = self.librarian.add_snippet(content, metadata=analysis)
                result = CommandResult(
                    True,
                    f"Ingested {path.name} as {sid[:8]}",
                    {"snippet_id": sid, "file": str(path)},
                )
        except Exception as e:
            result = CommandResult(False, f"Ingest error: {e}")
        self._output(result, args)

    def _cmd_library(self, args) -> None:
        """List code library contents."""
        try:
            snippets = list(self.librarian.snippets.items())[: args.limit]

            items = []
            for sid, snippet in snippets:
                items.append(
                    {
                        "id": sid[:8],
                        "preview": snippet.get("content", "")[:60] + "...",
                    }
                )

            result = CommandResult(
                True,
                f"Library contains {len(self.librarian.snippets)} snippets",
                {"snippets": items, "total": len(self.librarian.snippets)},
            )
        except Exception as e:
            result = CommandResult(False, f"Error: {e}")
        self._output(result, args)

    def _cmd_ingest_dir(self, args) -> None:
        """Ingest directory into SQLite-backed library."""
        try:
            path = Path(args.path)
            if not path.is_dir():
                result = CommandResult(False, "Path must be a directory")
            else:
                stats = self.runner.ingest_path(
                    path,
                    mode=args.mode,
                    extensions=args.ext,
                    max_files=args.max_files,
                    progress_every=args.progress_every,
                )
                manifest = self.runner.runs_dir / f"{stats.run_id}.json"
                result = CommandResult(
                    True,
                    f"Ingested {stats.files_processed} files ({stats.units_created} units)",
                    {
                        "run_id": stats.run_id,
                        "mode": stats.mode,
                        "files_processed": stats.files_processed,
                        "units_created": stats.units_created,
                        "errors": stats.errors,
                        "manifest": str(manifest),
                        "db_path": str(DEFAULT_DB_PATH),
                    },
                )
        except Exception as e:
            result = CommandResult(False, f"Ingest error: {e}")
        self._output(result, args)

    def _cmd_ingest_bg(self, args) -> None:
        """Start background ingestion process."""
        try:
            path = Path(args.path)
            if not path.is_dir():
                result = CommandResult(False, "Path must be a directory")
            else:
                import subprocess

                cmd = [
                    sys.executable,
                    "-m",
                    "code_forge.ingest.daemon",
                    "--root",
                    str(path),
                    "--mode",
                    args.mode,
                    "--progress-every",
                    str(args.progress_every),
                    "--detach",
                ]
                if args.ext:
                    cmd.extend(["--ext", *args.ext])
                if args.max_files is not None:
                    cmd.extend(["--max-files", str(args.max_files)])

                env = dict(**os.environ)
                env["PYTHONPATH"] = (
                    f"{FORGE_ROOT / 'code_forge' / 'src'}:{FORGE_ROOT / 'lib'}"
                )
                subprocess.Popen(cmd, env=env)

                result = CommandResult(
                    True,
                    "Background ingestion started",
                    {
                        "root": str(path),
                        "mode": args.mode,
                        "runs_dir": str(DEFAULT_RUNS_DIR),
                        "latest_run_file": str(DEFAULT_RUNS_DIR / "latest_run.json"),
                    },
                )
        except Exception as e:
            result = CommandResult(False, f"Ingest error: {e}")
        self._output(result, args)

    def _cmd_ingest_status(self, args) -> None:
        """Query progress from manifest."""
        try:
            manifest = DEFAULT_RUNS_DIR / f"{args.run_id}.json"
            if not manifest.exists():
                result = CommandResult(False, "Manifest not found")
            else:
                data = json.loads(manifest.read_text())
                result = CommandResult(True, "Status", data)
        except Exception as e:
            result = CommandResult(False, f"Status error: {e}")
        self._output(result, args)

    def _cmd_stats(self, args) -> None:
        """Show detailed statistics."""
        try:
            # Element types
            type_counts = {}
            file_counts = {}

            for elem in self.indexer.elements.values():
                t = elem.element_type
                type_counts[t] = type_counts.get(t, 0) + 1

                # Count files
                f = Path(elem.file_path).name
                file_counts[f] = file_counts.get(f, 0) + 1

            # Top files
            top_files = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[
                :10
            ]

            data = {
                "total_elements": len(self.indexer.elements),
                "by_type": type_counts,
                "unique_files": len(file_counts),
                "top_files": [{"file": f, "elements": c} for f, c in top_files],
                "library_size": len(self.librarian.snippets),
            }

            result = CommandResult(
                True,
                f"Stats: {data['total_elements']} elements, {data['unique_files']} files, {data['library_size']} snippets",
                data,
            )
        except Exception as e:
            result = CommandResult(False, f"Error: {e}")
        self._output(result, args)

    def _cmd_dedup_report(self, args) -> None:
        """Report duplicate code units."""
        try:
            groups = self.library_db.list_duplicate_units(
                min_occurrences=max(2, int(args.min_occurrences)),
                limit_groups=max(1, int(args.limit_groups)),
            )
            result = CommandResult(
                True,
                f"Found {len(groups)} duplicate groups",
                {
                    "duplicate_groups": groups,
                    "group_count": len(groups),
                    "db_path": str(DEFAULT_DB_PATH),
                },
            )
        except Exception as e:
            result = CommandResult(False, f"Dedup report error: {e}")
        self._output(result, args)

    def _cmd_normalized_dedup_report(self, args) -> None:
        """Report normalized duplicate units."""
        try:
            groups = self.library_db.list_normalized_duplicates(
                min_occurrences=max(2, int(args.min_occurrences)),
                limit_groups=max(1, int(args.limit_groups)),
            )
            result = CommandResult(
                True,
                f"Found {len(groups)} normalized duplicate groups",
                {
                    "duplicate_groups": groups,
                    "group_count": len(groups),
                    "db_path": str(DEFAULT_DB_PATH),
                },
            )
        except Exception as e:
            result = CommandResult(False, f"Normalized dedup report error: {e}")
        self._output(result, args)

    def _cmd_near_dedup_report(self, args) -> None:
        """Report near duplicate units by simhash distance."""
        try:
            pairs = self.library_db.list_near_duplicates(
                max_hamming=max(0, int(args.max_hamming)),
                min_token_count=max(0, int(args.min_tokens)),
                limit_pairs=max(1, int(args.limit_pairs)),
                language=args.language,
                unit_type=args.type,
            )
            result = CommandResult(
                True,
                f"Found {len(pairs)} near-duplicate pairs",
                {
                    "pairs": pairs,
                    "pair_count": len(pairs),
                    "filters": {
                        "max_hamming": int(args.max_hamming),
                        "min_tokens": int(args.min_tokens),
                        "language": args.language,
                        "unit_type": args.type,
                    },
                },
            )
        except Exception as e:
            result = CommandResult(False, f"Near dedup report error: {e}")
        self._output(result, args)

    def _cmd_semantic_search(self, args) -> None:
        """Run hybrid semantic search against indexed code units."""
        try:
            matches = self.library_db.semantic_search(
                query=str(args.query),
                limit=max(1, int(args.limit)),
                language=args.language,
                unit_type=args.type,
                min_score=float(args.min_score),
            )
            result = CommandResult(
                True,
                f"Found {len(matches)} semantic matches for '{args.query}'",
                {
                    "query": args.query,
                    "matches": matches,
                    "filters": {
                        "language": args.language,
                        "unit_type": args.type,
                        "min_score": float(args.min_score),
                    },
                },
            )
        except Exception as e:
            result = CommandResult(False, f"Semantic search error: {e}")
        self._output(result, args)

    def _cmd_trace(self, args) -> None:
        """Trace contains relationships from a code unit."""
        try:
            target = str(args.target).strip()
            root = self.library_db.get_unit(target)
            if root is None:
                root = self.library_db.find_unit_by_qualified_name(target)
            if root is None:
                result = CommandResult(False, f"Unit not found: {target}")
            else:
                trace = self.library_db.trace_contains(
                    str(root["id"]),
                    max_depth=max(1, int(args.depth)),
                    max_nodes=max(1, int(args.max_nodes)),
                )
                result = CommandResult(
                    True,
                    f"Trace built: {len(trace.get('nodes', []))} nodes, {len(trace.get('edges', []))} edges",
                    trace,
                )
        except Exception as e:
            result = CommandResult(False, f"Trace error: {e}")
        self._output(result, args)

    def _cmd_sync_knowledge(self, args) -> None:
        """Sync code units into Knowledge Forge."""
        try:
            payload = sync_units_to_knowledge_forge(
                db=self.library_db,
                kb_path=Path(args.kb_path),
                limit=max(1, int(args.limit)),
                min_token_count=max(0, int(args.min_tokens)),
                run_id=str(args.run_id).strip() if args.run_id else None,
                include_node_links=True,
                node_links_limit=100,
            )
            result = CommandResult(
                True,
                "Knowledge sync complete",
                payload,
            )
        except Exception as e:
            result = CommandResult(False, f"Knowledge sync error: {e}")
        self._output(result, args)

    def _cmd_export_graphrag(self, args) -> None:
        """Export code units as GraphRAG-ready text corpus."""
        try:
            payload = export_units_for_graphrag(
                db=self.library_db,
                output_dir=Path(args.output_dir),
                limit=max(1, int(args.limit)),
                min_token_count=max(0, int(args.min_tokens)),
                run_id=str(args.run_id).strip() if args.run_id else None,
            )
            result = CommandResult(
                True,
                f"Exported {payload.get('exported', 0)} code documents",
                payload,
            )
        except Exception as e:
            result = CommandResult(False, f"GraphRAG export error: {e}")
        self._output(result, args)

    def _cmd_catalog(self, args) -> None:
        """Build intake catalog + duplication artifacts."""
        try:
            root = Path(args.path).resolve()
            if not root.is_dir():
                result = CommandResult(False, f"Path must be a directory: {root}")
            else:
                output_dir = Path(args.output_dir).resolve()
                repo_index = build_repo_index(
                    root_path=root,
                    output_dir=output_dir,
                    extensions=args.ext,
                    max_files=args.max_files,
                )
                duplication = build_duplication_index(
                    db=self.library_db, output_dir=output_dir
                )
                result = CommandResult(
                    True,
                    f"Catalog complete for {repo_index.get('files_total', 0)} files",
                    {
                        "root_path": str(root),
                        "output_dir": str(output_dir),
                        "repo_index_path": str(output_dir / "repo_index.json"),
                        "duplication_index_path": str(
                            output_dir / "duplication_index.json"
                        ),
                        "repo_index_summary": {
                            "files_total": repo_index.get("files_total", 0),
                            "by_language": repo_index.get("by_language", {}),
                        },
                        "duplication_summary": duplication.get("summary", {}),
                    },
                )
        except Exception as e:
            result = CommandResult(False, f"Catalog error: {e}")
        self._output(result, args)

    def _cmd_triage_report(self, args) -> None:
        """Build explainable triage report using latest intake artifacts."""
        try:
            output_dir = Path(args.output_dir).resolve()
            repo_index_path = output_dir / "repo_index.json"
            duplication_path = output_dir / "duplication_index.json"

            if not repo_index_path.exists():
                result = CommandResult(
                    False, f"Missing intake artifact: {repo_index_path}"
                )
            elif not duplication_path.exists():
                result = CommandResult(
                    False, f"Missing intake artifact: {duplication_path}"
                )
            else:
                repo_index = json.loads(repo_index_path.read_text(encoding="utf-8"))
                duplication = json.loads(duplication_path.read_text(encoding="utf-8"))
                payload = build_triage_report(
                    db=self.library_db,
                    repo_index=repo_index,
                    duplication_index=duplication,
                    output_dir=output_dir,
                )
                result = CommandResult(
                    True,
                    "Triage report generated",
                    {
                        "output_dir": str(output_dir),
                        "triage_json": str(output_dir / "triage.json"),
                        "triage_csv": str(output_dir / "triage.csv"),
                        "triage_report": str(output_dir / "triage_report.md"),
                        "label_counts": payload.get("label_counts", {}),
                    },
                )
        except Exception as e:
            result = CommandResult(False, f"Triage error: {e}")
        self._output(result, args)

    def _cmd_digest(self, args) -> None:
        """Run archive digester end-to-end."""
        try:
            root = Path(args.path).resolve()
            if not root.is_dir():
                result = CommandResult(False, f"Path must be a directory: {root}")
            else:
                output_dir = Path(args.output_dir).resolve()
                kb_path = Path(args.kb_path).resolve() if args.sync_knowledge else None
                memory_path = (
                    Path(args.memory_path).resolve() if args.sync_memory else None
                )
                graphrag_out = (
                    Path(args.graphrag_output_dir).resolve()
                    if args.export_graphrag
                    else None
                )
                payload = run_archive_digester(
                    root_path=root,
                    db=self.library_db,
                    runner=self.runner,
                    output_dir=output_dir,
                    mode=args.mode,
                    extensions=args.ext,
                    max_files=args.max_files,
                    progress_every=max(1, int(args.progress_every)),
                    sync_knowledge_path=kb_path,
                    sync_memory_path=memory_path,
                    graphrag_output_dir=graphrag_out,
                    graph_export_limit=max(1, int(args.graph_export_limit)),
                    integration_policy=str(args.integration_policy),
                    strict_validation=not bool(args.no_strict_validation),
                    write_drift_report=not bool(args.no_drift_report),
                    write_history_snapshot=not bool(args.no_history_snapshot),
                    previous_snapshot_path=(
                        Path(args.previous_snapshot).resolve()
                        if args.previous_snapshot
                        else None
                    ),
                    drift_warn_pct=float(args.drift_warn_pct),
                    drift_min_abs_delta=float(args.drift_min_abs_delta),
                )
                result = CommandResult(
                    True,
                    "Archive digester completed",
                    payload,
                )
        except Exception as e:
            result = CommandResult(False, f"Digest error: {e}")
        self._output(result, args)

    def _cmd_dependency_graph(self, args) -> None:
        """Build aggregated dependency graph artifacts."""
        try:
            output_dir = Path(args.output_dir).resolve()
            payload = build_dependency_graph(
                db=self.library_db,
                output_dir=output_dir,
                rel_types=list(args.rel_types or []),
                limit_edges=max(1, int(args.limit_edges)),
            )
            result = CommandResult(
                True,
                "Dependency graph built",
                {
                    "output_dir": str(output_dir),
                    "dependency_graph_path": str(output_dir / "dependency_graph.json"),
                    "summary": payload.get("summary", {}),
                },
            )
        except Exception as e:
            result = CommandResult(False, f"Dependency graph error: {e}")
        self._output(result, args)

    def _cmd_validate_artifacts(self, args) -> None:
        """Validate digester artifacts against strict schema contracts."""
        try:
            output_dir = Path(args.output_dir).resolve()
            report = validate_output_dir(output_dir)
            if report.get("pass", False):
                result = CommandResult(
                    True,
                    "Artifact validation passed",
                    report,
                )
            else:
                result = CommandResult(
                    False,
                    "Artifact validation failed",
                    report,
                )
        except Exception as e:
            result = CommandResult(False, f"Artifact validation error: {e}")
        self._output(result, args)

    def _cmd_validate_roundtrip(self, args) -> None:
        """Validate roundtrip artifacts against strict contracts."""
        try:
            workspace_dir = Path(args.workspace_dir).resolve()
            report = validate_roundtrip_workspace(
                workspace_dir=workspace_dir,
                require_apply_report=bool(args.require_apply_report),
                verify_hashes=bool(args.verify_hashes),
            )
            if report.get("pass", False):
                result = CommandResult(True, "Roundtrip validation passed", report)
            else:
                result = CommandResult(False, "Roundtrip validation failed", report)
        except Exception as e:
            result = CommandResult(False, f"Roundtrip validation error: {e}")
        self._output(result, args)

    def _cmd_drift_report(self, args) -> None:
        """Generate a drift report using digester artifacts."""
        try:
            output_dir = Path(args.output_dir).resolve()
            report = build_drift_report_from_output(
                output_dir=output_dir,
                previous_snapshot_path=(
                    Path(args.previous_snapshot).resolve()
                    if args.previous_snapshot
                    else None
                ),
                history_dir=output_dir / "history",
                write_history=not bool(args.no_history_snapshot),
                warn_pct=float(args.drift_warn_pct),
                min_abs_delta=float(args.drift_min_abs_delta),
            )
            result = CommandResult(
                True,
                "Drift report generated",
                report,
            )
        except Exception as e:
            result = CommandResult(False, f"Drift report error: {e}")
        self._output(result, args)

    def _cmd_benchmark(self, args) -> None:
        """Run benchmark suite and optional regression gates."""
        try:
            output = Path(args.output).resolve()
            baseline = Path(args.baseline).resolve() if args.baseline else None
            queries = [q for q in (args.query or []) if str(q).strip()]
            payload = run_benchmark_suite(
                root_path=Path(args.root).resolve(),
                db_path=DEFAULT_DB_PATH,
                runs_dir=DEFAULT_RUNS_DIR,
                output_path=output,
                extensions=args.ext,
                max_files=args.max_files,
                ingestion_repeats=max(1, int(args.ingestion_repeats)),
                query_repeats=max(1, int(args.query_repeats)),
                queries=queries or None,
                baseline_path=baseline,
                max_regression_pct=float(args.max_regression_pct),
                prepare_ingest=not bool(args.skip_prepare_ingest),
                write_baseline=bool(args.write_baseline),
            )

            gate = payload.get("gate", {})
            message = "Benchmark suite complete"
            if gate.get("baseline_loaded") and not gate.get("pass", False):
                message = "Benchmark suite complete (regression gate failed)"

            result = CommandResult(
                True,
                message,
                {
                    "output": str(output),
                    "baseline": str(baseline) if baseline else None,
                    "gate": gate,
                    "ingestion": payload.get("ingestion", {}),
                    "search": payload.get("search", {}),
                    "graph": payload.get("graph", {}),
                },
            )
        except Exception as e:
            result = CommandResult(False, f"Benchmark error: {e}")
        self._output(result, args)

    def _cmd_eval_init(self, args) -> None:
        """Create sample eval taskbank and config matrix files."""
        try:
            taskbank_path = Path(args.taskbank).resolve()
            matrix_path = Path(args.matrix).resolve()
            if not args.force:
                existing = [p for p in (taskbank_path, matrix_path) if p.exists()]
                if existing:
                    result = CommandResult(
                        False,
                        "Refusing to overwrite existing eval files without --force",
                        {"existing": [str(p) for p in existing]},
                    )
                    self._output(result, args)
                    return

            taskbank = create_sample_taskbank(taskbank_path)
            matrix = create_sample_config_matrix(matrix_path)
            result = CommandResult(
                True,
                "Eval sample configuration files created",
                {
                    "taskbank_path": str(taskbank_path),
                    "matrix_path": str(matrix_path),
                    "task_count": len(taskbank.get("tasks") or []),
                    "config_count": len(matrix.get("configs") or []),
                },
            )
        except Exception as e:
            result = CommandResult(False, f"Eval init error: {e}")
        self._output(result, args)

    def _cmd_eval_run(self, args) -> None:
        """Execute taskbank x config matrix evaluation runs."""
        try:
            options = EvalRunOptions(
                taskbank_path=Path(args.taskbank).resolve(),
                config_matrix_path=Path(args.matrix).resolve(),
                output_dir=Path(args.output_dir).resolve(),
                repo_root=Path(args.repo_root).resolve(),
                repeats=max(1, int(args.repeats)),
                replay_mode=str(args.replay_mode),
                max_parallel=max(1, int(args.max_parallel)),
                default_timeout_sec=max(1, int(args.default_timeout_sec)),
                staleness_log_path=(
                    Path(args.staleness_log).resolve() if args.staleness_log else None
                ),
            )
            payload = run_eval_suite(options)
            run_stats = payload.get("run_stats", {})
            result = CommandResult(
                True,
                "Eval suite complete",
                {
                    "output_dir": str(options.output_dir),
                    "summary_path": str(options.output_dir / "summary.json"),
                    "run_count": run_stats.get("run_count", 0),
                    "success_rate": run_stats.get("success_rate", 0.0),
                    "replay_hits": run_stats.get("replay_hits", 0),
                    "replay_misses": run_stats.get("replay_misses", 0),
                    "config_scores": payload.get("config_scores", {}),
                },
            )
        except Exception as e:
            result = CommandResult(False, f"Eval run error: {e}")
        self._output(result, args)

    def _cmd_eval_staleness(self, args) -> None:
        """Compute staleness/freshness metrics from a log payload."""
        try:
            input_path = Path(args.input).resolve()
            records = load_freshness_records(input_path)
            payload = compute_staleness_metrics(records)
            if args.output:
                output_path = Path(args.output).resolve()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
                )
                payload = dict(payload)
                payload["output_path"] = str(output_path)

            result = CommandResult(
                True,
                "Staleness metrics computed",
                payload,
            )
        except Exception as e:
            result = CommandResult(False, f"Eval staleness error: {e}")
        self._output(result, args)

    def _cmd_canonical_plan(self, args) -> None:
        """Generate canonical migration map and shim staging artifacts."""
        try:
            payload = build_canonical_migration_plan(
                triage_path=Path(args.triage_path).resolve(),
                output_dir=Path(args.output_dir).resolve(),
                include_labels=list(args.labels or []),
                max_entries=max(1, int(args.max_entries)),
                generate_shims=not bool(args.no_shims),
            )
            result = CommandResult(
                True,
                "Canonical migration plan generated",
                payload,
            )
        except Exception as e:
            result = CommandResult(False, f"Canonical plan error: {e}")
        self._output(result, args)

    def _cmd_reconstruct_project(self, args) -> None:
        """Reconstruct project files from library file records/content blobs."""
        try:
            payload = build_reconstruction_from_library(
                db=self.library_db,
                source_root=Path(args.source_root).resolve(),
                output_dir=Path(args.output_dir).resolve(),
                extensions=args.ext,
                max_files=args.max_files,
                strict=not bool(args.no_strict),
            )
            result = CommandResult(
                True,
                f"Reconstructed {payload.get('files_written', 0)} files",
                payload,
            )
        except Exception as e:
            result = CommandResult(False, f"Reconstruction error: {e}")
        self._output(result, args)

    def _cmd_parity_report(self, args) -> None:
        """Generate source vs reconstructed tree parity report."""
        try:
            report_path = Path(args.report_path).resolve() if args.report_path else None
            payload = compare_tree_parity(
                source_root=Path(args.source_root).resolve(),
                reconstructed_root=Path(args.reconstructed_root).resolve(),
                report_path=report_path,
                extensions=args.ext,
            )
            result = CommandResult(
                bool(payload.get("pass")),
                "Parity check passed" if payload.get("pass") else "Parity check failed",
                payload,
            )
        except Exception as e:
            result = CommandResult(False, f"Parity report error: {e}")
        self._output(result, args)

    def _cmd_apply_reconstruction(self, args) -> None:
        """Apply reconstructed tree to target root with backups."""
        try:
            parity_payload = None
            if args.parity_report:
                parity_payload = json.loads(
                    Path(args.parity_report).resolve().read_text(encoding="utf-8")
                )
            payload = apply_reconstruction(
                reconstructed_root=Path(args.reconstructed_root).resolve(),
                target_root=Path(args.target_root).resolve(),
                backup_root=Path(args.backup_root).resolve(),
                parity_report=parity_payload,
                require_parity_pass=not bool(args.allow_non_parity),
                prune=not bool(args.no_prune),
                require_manifest=bool(args.require_manifest),
                dry_run=bool(args.dry_run),
            )
            result = CommandResult(
                True,
                (
                    "Reconstruction apply dry-run complete"
                    if payload.get("dry_run")
                    else (
                        "Reconstruction applied"
                        if not payload.get("noop")
                        else "Reconstruction apply no-op (already in sync)"
                    )
                ),
                payload,
            )
        except Exception as e:
            result = CommandResult(False, f"Apply reconstruction error: {e}")
        self._output(result, args)

    def _cmd_roundtrip(self, args) -> None:
        """Run full digest + reconstruction + parity (+optional apply) pipeline."""
        try:
            root = Path(args.path).resolve()
            if not root.is_dir():
                result = CommandResult(False, f"Path must be a directory: {root}")
            else:
                payload = run_roundtrip_pipeline(
                    root_path=root,
                    db=self.library_db,
                    runner=self.runner,
                    workspace_dir=Path(args.workspace_dir).resolve(),
                    mode=args.mode,
                    extensions=args.ext,
                    max_files=args.max_files,
                    progress_every=max(1, int(args.progress_every)),
                    sync_knowledge_path=(
                        Path(args.kb_path).resolve() if args.sync_knowledge else None
                    ),
                    sync_memory_path=(
                        Path(args.memory_path).resolve() if args.sync_memory else None
                    ),
                    graphrag_output_dir=(
                        Path(args.graphrag_output_dir).resolve()
                        if args.export_graphrag
                        else None
                    ),
                    graph_export_limit=max(1, int(args.graph_export_limit)),
                    integration_policy=str(args.integration_policy),
                    strict_validation=not bool(args.no_strict_validation),
                    apply=bool(args.apply),
                    backup_root=Path(args.backup_root).resolve(),
                )
                result = CommandResult(
                    bool(payload.get("parity_pass")),
                    "Roundtrip pipeline complete",
                    payload,
                )
        except Exception as e:
            result = CommandResult(False, f"Roundtrip error: {e}")
        self._output(result, args)


@eidosian()
def main():
    """Entry point for code-forge CLI."""
    cli = CodeForgeCLI()
    sys.exit(cli.run())


if __name__ == "__main__":
    main()
