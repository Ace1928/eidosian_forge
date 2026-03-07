"""Archive digestion pipeline for Code Forge."""

from code_forge.digester.drift import build_drift_report_from_output
from code_forge.digester.pipeline import (
    build_archive_ingestion_batches,
    build_dependency_graph,
    build_duplication_index,
    build_repo_index,
    build_triage_report,
    initialize_archive_ingestion_state,
    load_archive_ingestion_state,
    run_archive_digester,
    run_archive_ingestion_batches,
    update_archive_ingestion_state,
)
from code_forge.digester.schema import validate_output_dir

__all__ = [
    "build_dependency_graph",
    "build_archive_ingestion_batches",
    "build_duplication_index",
    "build_repo_index",
    "build_triage_report",
    "build_drift_report_from_output",
    "initialize_archive_ingestion_state",
    "load_archive_ingestion_state",
    "run_archive_digester",
    "run_archive_ingestion_batches",
    "update_archive_ingestion_state",
    "validate_output_dir",
]
