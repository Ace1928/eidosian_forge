import os
import sys
from pathlib import Path

# --- Root Configuration ---
FORGE_ROOT = Path(os.environ.get("EIDOS_FORGE_ROOT", "/data/data/com.termux/files/home/eidosian_forge")).resolve()
HOME_ROOT = Path(os.environ.get("HOME", "/data/data/com.termux/files/home")).resolve()
RUNTIME_DIR = FORGE_ROOT / "data" / "runtime"

# --- Import Paths ---
FORGE_IMPORT_PATHS = (
    FORGE_ROOT / "atlas_forge" / "src",
    FORGE_ROOT / "web_interface_forge" / "src",
    FORGE_ROOT / "lib",
    FORGE_ROOT / "doc_forge" / "src",
    FORGE_ROOT / "agent_forge" / "src",
    FORGE_ROOT / "code_forge" / "src",
    FORGE_ROOT / "file_forge" / "src",
    FORGE_ROOT / "gis_forge" / "src",
    FORGE_ROOT / "memory_forge" / "src",
    FORGE_ROOT / "knowledge_forge" / "src",
    FORGE_ROOT / "word_forge" / "src",
    FORGE_ROOT / "narrative_forge" / "src",
    FORGE_ROOT / "eidos_mcp" / "src",
    FORGE_ROOT / "neural_forge" / "src",
)

def setup_pythonpath():
    for extra in FORGE_IMPORT_PATHS:
        extra_text = str(extra)
        if extra.exists() and extra_text not in sys.path:
            sys.path.insert(0, extra_text)

# --- Documentation Paths ---
DOC_RUNTIME = FORGE_ROOT / "doc_forge" / "runtime"
DOC_FINAL = DOC_RUNTIME / "final_docs"
DOC_INDEX = DOC_RUNTIME / "doc_index.json"
DOC_STATUS = DOC_RUNTIME / "processor_status.json"
DOC_HISTORY = DOC_RUNTIME / "processor_history.jsonl"

# --- Runtime Status Paths ---
LOCAL_AGENT_STATUS = RUNTIME_DIR / "local_mcp_agent" / "status.json"
LOCAL_AGENT_HISTORY = RUNTIME_DIR / "local_mcp_agent" / "history.jsonl"
QWENCHAT_STATUS = RUNTIME_DIR / "qwenchat" / "status.json"
QWENCHAT_HISTORY = RUNTIME_DIR / "qwenchat" / "history.jsonl"
SCHEDULER_STATUS = RUNTIME_DIR / "eidos_scheduler_status.json"
SCHEDULER_HISTORY = RUNTIME_DIR / "eidos_scheduler_history.jsonl"
LIVING_PIPELINE_STATUS = RUNTIME_DIR / "living_pipeline_status.json"
LIVING_PIPELINE_HISTORY = RUNTIME_DIR / "living_pipeline_history.jsonl"
COORDINATOR_STATUS = RUNTIME_DIR / "forge_coordinator_status.json"
COORDINATOR_HISTORY = RUNTIME_DIR / "forge_runtime_trends.json"
BOOT_STATUS = RUNTIME_DIR / "termux_boot_status.json"
CAPABILITIES_STATUS = RUNTIME_DIR / "platform_capabilities.json"
DIRECTORY_DOCS_STATUS = RUNTIME_DIR / "directory_docs_status.json"
DIRECTORY_DOCS_HISTORY = RUNTIME_DIR / "directory_docs_history.json"
DIRECTORY_DOCS_TREE = RUNTIME_DIR / "directory_docs_tree.json"
DOCS_BATCH_STATUS = RUNTIME_DIR / "docs_upsert_batch_status.json"
DOCS_BATCH_HISTORY = RUNTIME_DIR / "docs_upsert_batch_history.jsonl"
PROOF_REFRESH_STATUS = RUNTIME_DIR / "proof_refresh_status.json"
PROOF_REFRESH_HISTORY = RUNTIME_DIR / "proof_refresh_history.jsonl"
RUNTIME_BENCHMARK_RUN_STATUS = RUNTIME_DIR / "runtime_benchmark_run_status.json"
RUNTIME_BENCHMARK_RUN_HISTORY = RUNTIME_DIR / "runtime_benchmark_run_history.jsonl"
RUNTIME_ARTIFACT_AUDIT_STATUS = RUNTIME_DIR / "runtime_artifact_audit_status.json"
RUNTIME_ARTIFACT_AUDIT_HISTORY = RUNTIME_DIR / "runtime_artifact_audit_history.jsonl"
CODE_FORGE_PROVENANCE_AUDIT_STATUS = RUNTIME_DIR / "code_forge_provenance_audit_status.json"
CODE_FORGE_PROVENANCE_AUDIT_HISTORY = RUNTIME_DIR / "code_forge_provenance_audit_history.jsonl"
CODE_FORGE_ARCHIVE_PLAN_STATUS = RUNTIME_DIR / "code_forge_archive_plan_status.json"
CODE_FORGE_ARCHIVE_PLAN_HISTORY = RUNTIME_DIR / "code_forge_archive_plan_history.jsonl"
CODE_FORGE_ARCHIVE_LIFECYCLE_STATUS = RUNTIME_DIR / "code_forge_archive_lifecycle_status.json"
CODE_FORGE_ARCHIVE_LIFECYCLE_HISTORY = RUNTIME_DIR / "code_forge_archive_lifecycle_history.jsonl"
FILE_FORGE_INDEX_STATUS = RUNTIME_DIR / "file_forge_index_status.json"
FILE_FORGE_INDEX_HISTORY = RUNTIME_DIR / "file_forge_index_history.jsonl"
FILE_FORGE_DB = FORGE_ROOT / "data" / "file_forge" / "library.sqlite"
WORD_FORGE_DB = FORGE_ROOT / "word_forge" / "data" / "word_forge.sqlite"
WORD_FORGE_MULTILINGUAL_INGEST_STATUS = RUNTIME_DIR / "word_forge_multilingual_ingest_status.json"
WORD_FORGE_MULTILINGUAL_INGEST_HISTORY = RUNTIME_DIR / "word_forge_multilingual_ingest_history.jsonl"
WORD_FORGE_BRIDGE_AUDIT_STATUS = RUNTIME_DIR / "word_forge_bridge_audit_status.json"
WORD_FORGE_BRIDGE_AUDIT_HISTORY = RUNTIME_DIR / "word_forge_bridge_audit_history.jsonl"

# --- Bridge Paths ---
SESSION_BRIDGE_DIR = RUNTIME_DIR / "session_bridge"
SESSION_BRIDGE_CONTEXT = SESSION_BRIDGE_DIR / "latest_context.json"
SESSION_BRIDGE_IMPORT_STATUS = SESSION_BRIDGE_DIR / "import_status.json"

# --- Report Paths ---
PROOF_REPORT_DIR = FORGE_ROOT / "reports" / "proof"
PROOF_BUNDLE_DIR = FORGE_ROOT / "reports" / "proof_bundle"
SECURITY_REPORT_DIR = FORGE_ROOT / "reports" / "security"
RUNTIME_ARTIFACT_REPORT_DIR = FORGE_ROOT / "reports" / "runtime_artifact_audit"
CODE_FORGE_PROVENANCE_REPORT_DIR = FORGE_ROOT / "reports" / "code_forge_provenance_audit"
CODE_FORGE_ARCHIVE_PLAN_REPORT_DIR = FORGE_ROOT / "reports" / "code_forge_archive_plan"
CODE_FORGE_ARCHIVE_LIFECYCLE_REPORT_DIR = FORGE_ROOT / "reports" / "code_forge_archive_lifecycle"
CODE_FORGE_ARCHIVE_RETIREMENTS_LATEST = FORGE_ROOT / "data" / "code_forge" / "archive_ingestion" / "latest" / "retirements" / "latest.json"
WORD_FORGE_MULTILINGUAL_REPORT_DIR = FORGE_ROOT / "reports" / "word_forge_multilingual_ingest"
WORD_FORGE_BRIDGE_REPORT_DIR = FORGE_ROOT / "reports" / "word_forge_bridge_audit"

# --- Scripts & Logs ---
SERVICES_SCRIPT = FORGE_ROOT / "scripts" / "eidos_termux_services.sh"
SERVICE_ACTION_LOG = RUNTIME_DIR / "atlas_service_actions.log"
SCHEDULER_CONTROL_SCRIPT = FORGE_ROOT / "scripts" / "eidos_scheduler_control.py"

# --- App Layout ---
TEMPLATES_DIR = Path(__file__).parents[2] / "templates"
STATIC_DIR = Path(__file__).parents[2] / "static"
