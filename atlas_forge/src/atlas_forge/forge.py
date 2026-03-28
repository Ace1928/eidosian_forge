import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from dataclasses import asdict
from eidosian_runtime import collect_runtime_capabilities

from .config import (
    FORGE_ROOT, DOC_STATUS, DOC_INDEX, DOC_HISTORY, 
    FILE_FORGE_INDEX_STATUS, FILE_FORGE_INDEX_HISTORY, FILE_FORGE_DB,
    COORDINATOR_STATUS, SCHEDULER_STATUS, SCHEDULER_HISTORY,
    LOCAL_AGENT_STATUS, LOCAL_AGENT_HISTORY,
    QWENCHAT_STATUS, QWENCHAT_HISTORY,
    LIVING_PIPELINE_STATUS, LIVING_PIPELINE_HISTORY,
    BOOT_STATUS, 
    CAPABILITIES_STATUS, DIRECTORY_DOCS_STATUS, DIRECTORY_DOCS_HISTORY,
    DIRECTORY_DOCS_TREE, SESSION_BRIDGE_CONTEXT, SESSION_BRIDGE_IMPORT_STATUS,
    PROOF_REPORT_DIR, PROOF_BUNDLE_DIR, SECURITY_REPORT_DIR,
    RUNTIME_ARTIFACT_REPORT_DIR, CODE_FORGE_PROVENANCE_REPORT_DIR,
    CODE_FORGE_ARCHIVE_PLAN_REPORT_DIR, CODE_FORGE_ARCHIVE_LIFECYCLE_REPORT_DIR,
    CODE_FORGE_ARCHIVE_RETIREMENTS_LATEST, DOCS_BATCH_STATUS, DOCS_BATCH_HISTORY,
    PROOF_REFRESH_STATUS, PROOF_REFRESH_HISTORY, RUNTIME_BENCHMARK_RUN_STATUS,
    RUNTIME_BENCHMARK_RUN_HISTORY, RUNTIME_ARTIFACT_AUDIT_STATUS,
    RUNTIME_ARTIFACT_AUDIT_HISTORY, CODE_FORGE_PROVENANCE_AUDIT_STATUS,
    CODE_FORGE_PROVENANCE_AUDIT_HISTORY, CODE_FORGE_ARCHIVE_PLAN_STATUS,
    CODE_FORGE_ARCHIVE_PLAN_HISTORY, CODE_FORGE_ARCHIVE_LIFECYCLE_STATUS,
    CODE_FORGE_ARCHIVE_LIFECYCLE_HISTORY, HOME_ROOT, WORD_FORGE_DB,
    WORD_FORGE_MULTILINGUAL_INGEST_STATUS, WORD_FORGE_MULTILINGUAL_INGEST_HISTORY,
    WORD_FORGE_BRIDGE_AUDIT_STATUS, WORD_FORGE_BRIDGE_AUDIT_HISTORY,
    WORD_FORGE_MULTILINGUAL_REPORT_DIR, WORD_FORGE_BRIDGE_REPORT_DIR
)
from .utils import _read_json, _read_jsonl_rows, _resolve_operator_path
from .word_graph import build_word_graph_neighbor_payload, build_word_graph_payload, summarize_word_graph_communities

logger = logging.getLogger("eidos_dashboard")

# --- Helpers ---

def _get_ts(dt: Any) -> datetime:
    if not isinstance(dt, datetime):
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt

# --- Lazy Imports ---
def get_file_library_db():
    try:
        from file_forge.library import FileLibraryDB
        return FileLibraryDB
    except ImportError:
        logger.error("file_forge.library not found")
        return None

# --- Forge Status & Snapshots ---

def get_forge_status() -> Dict[str, Any]:
    status = {"doc_forge": "unknown", "details": {}}
    if DOC_STATUS.exists():
        try:
            data = json.loads(DOC_STATUS.read_text())
            status["doc_forge"] = data.get("status", "unknown")
            status["details"] = data
        except Exception:
            pass
    return status

def get_doc_snapshot() -> Dict[str, Any]:
    status_payload = _read_json(DOC_STATUS, {})
    index_payload = _read_json(DOC_INDEX, {})
    entries = index_payload.get("entries", [])
    if not isinstance(entries, list):
        entries = []

    recent_docs = [entry for entry in entries if isinstance(entry, dict)]
    recent_docs.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    recent_docs = recent_docs[:12]

    return {
        "status": status_payload,
        "index_count": len(entries),
        "recent_docs": recent_docs,
    }

def get_file_forge_summary(path_prefix: str = "", recent_limit: int = 8) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "contract": "eidos.file_forge.summary.v1",
        "status": "missing",
        "db_path": str(FILE_FORGE_DB),
        "path_prefix": path_prefix or None,
        "latest_index": _read_json(FILE_FORGE_INDEX_STATUS, {"status": "idle"}),
        "recent_files": [],
        "by_kind": [],
        "by_forge": [],
    }
    if not FILE_FORGE_DB.exists():
        return payload
        
    FileLibraryDB = get_file_library_db()
    if not FileLibraryDB:
        payload["status"] = "error"
        payload["error"] = "FileLibraryDB unavailable"
        return payload
        
    try:
        summary = FileLibraryDB(FILE_FORGE_DB).summary(
            path_prefix=_resolve_operator_path(path_prefix, FORGE_ROOT, HOME_ROOT) if path_prefix else None,
            recent_limit=recent_limit,
        )
        for row in summary.get("recent_files", []):
            file_path = Path(str(row.get("file_path") or ""))
            try:
                row["file_path"] = str(file_path.relative_to(FORGE_ROOT))
            except Exception:
                row["file_path"] = str(file_path)
        payload.update(summary)
        payload["status"] = "ready"
        payload["db_exists"] = True
    except Exception as e:
        payload["status"] = "error"
        payload["error"] = str(e)
        
    return payload

def _read_latest_report(report_dir: Path) -> Dict[str, Any]:
    return _read_json(report_dir / "latest.json", {})

def get_word_forge_multilingual_summary() -> Dict[str, Any]:
    latest = _read_latest_report(WORD_FORGE_MULTILINGUAL_REPORT_DIR)
    return {
        "contract": "eidos.word_forge.multilingual.summary.v1",
        "status": _read_json(WORD_FORGE_MULTILINGUAL_INGEST_STATUS, {"status": "idle"}),
        "history": _read_jsonl_rows(WORD_FORGE_MULTILINGUAL_INGEST_HISTORY, 12),
        "latest_report": latest,
        "db_path": str(WORD_FORGE_DB),
    }

def get_word_forge_bridge_summary() -> Dict[str, Any]:
    latest = _read_latest_report(WORD_FORGE_BRIDGE_REPORT_DIR)
    history = _read_jsonl_rows(WORD_FORGE_BRIDGE_AUDIT_HISTORY, 12)
    community_summary = summarize_word_graph_communities(get_word_graph(), limit=8)
    latest_completed = next((row for row in reversed(history) if isinstance(row, dict) and row.get("status") == "completed"), {})
    previous_completed = next((row for row in reversed(history[:-1]) if isinstance(row, dict) and row.get("status") == "completed"), {}) if len(history) > 1 else {}
    latest_report = latest if isinstance(latest, dict) else {}
    bridge_counts = latest_report.get("bridge_counts", {}) if isinstance(latest_report.get("bridge_counts"), dict) else {}
    bridge_quality = latest_report.get("bridge_quality", {}) if isinstance(latest_report.get("bridge_quality"), dict) else {}
    return {
        "contract": "eidos.word_forge.bridge.summary.v1",
        "status": _read_json(WORD_FORGE_BRIDGE_AUDIT_STATUS, {"status": "idle"}),
        "history": history,
        "latest_report": latest_report,
        "bridge_counts": bridge_counts,
        "bridge_quality": bridge_quality,
        "history_summary": {
            "run_count": len(history),
            "last_completed_at": latest_completed.get("finished_at") or latest_completed.get("generated_at") or "",
            "previous_completed_at": previous_completed.get("finished_at") or previous_completed.get("generated_at") or "",
        },
        "community_summary": community_summary,
    }

def get_docs_history(limit: int = 60) -> List[Dict[str, Any]]:
    payload = _read_json(DIRECTORY_DOCS_HISTORY, {})
    rows = payload.get("entries", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)][-max(1, int(limit)) :]

def get_file_forge_index_history(limit: int = 12) -> List[Dict[str, Any]]:
    return _read_jsonl_rows(FILE_FORGE_INDEX_HISTORY, limit)

def get_word_forge_multilingual_history(limit: int = 12) -> List[Dict[str, Any]]:
    return _read_jsonl_rows(WORD_FORGE_MULTILINGUAL_INGEST_HISTORY, limit)

def get_word_forge_bridge_history(limit: int = 12) -> List[Dict[str, Any]]:
    return _read_jsonl_rows(WORD_FORGE_BRIDGE_AUDIT_HISTORY, limit)

def get_word_graph_communities(limit: int = 12) -> Dict[str, Any]:
    return summarize_word_graph_communities(get_word_graph(), limit=limit)

def get_local_agent_history(limit: int = 12) -> List[Dict[str, Any]]:
    return _read_jsonl_rows(LOCAL_AGENT_HISTORY, limit)

def get_qwenchat_history(limit: int = 12) -> List[Dict[str, Any]]:
    return _read_jsonl_rows(QWENCHAT_HISTORY, limit)

def get_living_pipeline_history(limit: int = 12) -> List[Dict[str, Any]]:
    return _read_jsonl_rows(LIVING_PIPELINE_HISTORY, limit)

def get_doc_processor_history(limit: int = 12) -> List[Dict[str, Any]]:
    return _read_jsonl_rows(DOC_HISTORY, limit)

def get_scheduler_history(limit: int = 12) -> List[Dict[str, Any]]:
    return _read_jsonl_rows(SCHEDULER_HISTORY, limit)

def get_docs_batch_history(limit: int = 12) -> List[Dict[str, Any]]:
    return _read_jsonl_rows(DOCS_BATCH_HISTORY, limit)

def get_proof_refresh_history(limit: int = 12) -> List[Dict[str, Any]]:
    return _read_jsonl_rows(PROOF_REFRESH_HISTORY, limit)

def get_runtime_benchmark_run_history(limit: int = 12) -> List[Dict[str, Any]]:
    return _read_jsonl_rows(RUNTIME_BENCHMARK_RUN_HISTORY, limit)

def get_runtime_artifact_audit_history(limit: int = 12) -> List[Dict[str, Any]]:
    return _read_jsonl_rows(RUNTIME_ARTIFACT_AUDIT_HISTORY, limit)

def get_code_forge_provenance_audit_history(limit: int = 12) -> List[Dict[str, Any]]:
    return _read_jsonl_rows(CODE_FORGE_PROVENANCE_AUDIT_HISTORY, limit)

def get_code_forge_archive_plan_history(limit: int = 12) -> List[Dict[str, Any]]:
    return _read_jsonl_rows(CODE_FORGE_ARCHIVE_PLAN_HISTORY, limit)

def get_code_forge_archive_lifecycle_history(limit: int = 12) -> List[Dict[str, Any]]:
    return _read_jsonl_rows(CODE_FORGE_ARCHIVE_LIFECYCLE_HISTORY, limit)

def get_session_bridge_status() -> Dict[str, Any]:
    payload = {
        "contract": "eidos.session_bridge.status.v1",
        "context": _read_json(SESSION_BRIDGE_CONTEXT, {}),
        "import_status": _read_json(SESSION_BRIDGE_IMPORT_STATUS, {}),
    }
    try:
        from eidosian_runtime.session_bridge import recent_session_digest, summarize_import_status
        payload["recent_sessions"] = recent_session_digest(limit=6)
        payload["summary"] = summarize_import_status(payload.get("import_status"))
    except Exception as exc:
        payload["recent_sessions"] = []
        payload["summary"] = {}
        payload["error"] = str(exc)
    return payload

def get_proof_summary() -> Dict[str, Any]:
    proof = _read_json(PROOF_REPORT_DIR / "entity_proof_scorecard_latest.json", {})
    bundle = _read_json(PROOF_BUNDLE_DIR / "latest_manifest.json", {})
    identity = _read_json(PROOF_REPORT_DIR / "identity_continuity_scorecard_latest.json", {})
    session_bridge = get_session_bridge_status()
    
    # Get security reports
    security = {}
    for path in sorted(SECURITY_REPORT_DIR.glob("dependabot_open_summary_*.json"), reverse=True):
        security = _read_json(path, {})
        if security: break
        
    security_plan = {}
    for path in sorted(SECURITY_REPORT_DIR.glob("dependabot_remediation_plan_*.json"), reverse=True):
        security_plan = _read_json(path, {})
        if security_plan: break

    history = identity.get("history") if isinstance(identity.get("history"), dict) else {}
    
    return {
        "contract": "eidos.proof.summary.v1",
        "proof": proof,
        "bundle": bundle,
        "identity": identity,
        "identity_history": get_identity_continuity_history(limit=12),
        "proof_history": get_proof_history(limit=12),
        "external_benchmarks": get_external_benchmark_results(limit=12),
        "runtime_benchmarks": get_runtime_benchmark_statuses(limit=12),
        "security": {
            "summary": security,
            "plan": security_plan,
        },
        "session_bridge": {
            "recent_sessions": len(session_bridge.get("recent_sessions") or []) if isinstance(session_bridge.get("recent_sessions"), list) else 0,
            "last_sync_at": ((session_bridge.get("summary") or {}).get("last_sync_at")),
            "codex_records": ((session_bridge.get("summary") or {}).get("codex_records", 0)),
            "gemini_records": ((session_bridge.get("summary") or {}).get("gemini_records", 0)),
            "imported_records": ((session_bridge.get("summary") or {}).get("imported_records", 0)),
        },
        "identity_trend": history.get("trend"),
        "identity_delta": history.get("delta_from_previous"),
    }

def get_identity_continuity_history(limit: int = 12) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(PROOF_REPORT_DIR.glob("identity_continuity_scorecard_*.json"), reverse=True):
        if path.name.endswith("_latest.json"): continue
        payload = _read_json(path, {})
        if not payload: continue
        rows.append({
            "generated_at": payload.get("generated_at") or payload.get("ts") or "",
            "overall_score": payload.get("overall_score"),
            "status": payload.get("status", ""),
            "recent_sessions": ((payload.get("session_bridge") or {}).get("recent_sessions", 0)),
            "path": str(path.relative_to(FORGE_ROOT)),
        })
        if len(rows) >= limit: break
    return list(reversed(rows))

def get_proof_history(limit: int = 12) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(PROOF_REPORT_DIR.glob("entity_proof_scorecard_*.json"), reverse=True):
        if path.name.endswith("_latest.json"): continue
        payload = _read_json(path, {})
        if not payload: continue
        overall = payload.get("overall") if isinstance(payload.get("overall"), dict) else {}
        freshness = payload.get("freshness") if isinstance(payload.get("freshness"), dict) else {}
        regression = payload.get("regression") if isinstance(payload.get("regression"), dict) else {}
        rows.append({
            "generated_at": payload.get("generated_at") or "",
            "overall_score": overall.get("score"),
            "status": overall.get("status", ""),
            "freshness_status": freshness.get("status", ""),
            "regression_status": regression.get("status", ""),
            "path": str(path.relative_to(FORGE_ROOT)),
        })
        if len(rows) >= limit: break
    return list(reversed(rows))

def get_external_benchmark_results(limit: int = 12) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    root = FORGE_ROOT / "reports" / "external_benchmarks"
    if not root.exists(): return rows
    for latest in sorted(root.glob("*/latest.json")):
        payload = _read_json(latest, {})
        if not payload: continue
        rows.append({
            "suite": payload.get("suite") or latest.parent.name,
            "score": payload.get("score"),
            "status": payload.get("status", ""),
            "participant": payload.get("participant", ""),
            "execution_mode": payload.get("execution_mode", ""),
            "generated_at": payload.get("generated_at", ""),
            "path": str(latest.relative_to(FORGE_ROOT)),
        })
    rows.sort(key=lambda row: str(row.get("generated_at") or ""), reverse=True)
    return rows[:limit]

def get_runtime_benchmark_statuses(limit: int = 12) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    root = FORGE_ROOT / "data" / "runtime" / "external_benchmarks" / "agencybench"
    if not root.exists(): return rows
    for status_path in sorted(root.glob("**/status.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        payload = _read_json(status_path, {})
        if not payload: continue
        rows.append({
            "scenario": payload.get("scenario") or status_path.parent.name,
            "engine": payload.get("engine", ""),
            "model": payload.get("model", ""),
            "status": payload.get("status", ""),
            "stop_reason": payload.get("stop_reason", ""),
            "completed_count": payload.get("completed_count", 0),
            "attempt_count": payload.get("attempt_count", 0),
            "generated_at": payload.get("generated_at", ""),
            "path": str(status_path.relative_to(FORGE_ROOT)),
            "run_root": payload.get("run_root", ""),
        })
        if len(rows) >= limit: break
    return rows

def get_runtime_snapshot() -> Dict[str, Any]:
    from .shell import _shell_sessions_snapshot
    coordinator = _read_json(COORDINATOR_STATUS, {})
    scheduler = _read_json(SCHEDULER_STATUS, {})
    local_agent = _read_json(LOCAL_AGENT_STATUS, {})
    qwenchat = _read_json(QWENCHAT_STATUS, {})
    living_pipeline = _read_json(LIVING_PIPELINE_STATUS, {})
    doc_processor = _read_json(DOC_STATUS, {})
    file_forge = get_file_forge_summary(recent_limit=6)
    boot_status = _read_json(BOOT_STATUS, {})
    capabilities = _read_json(CAPABILITIES_STATUS, {})
    if not capabilities:
        capabilities = asdict(collect_runtime_capabilities())
    
    directory_docs = _read_json(DIRECTORY_DOCS_STATUS, {})
    proof_summary = get_proof_summary()
    
    return {
        "coordinator": coordinator,
        "scheduler": scheduler,
        "local_agent": local_agent,
        "qwenchat": qwenchat,
        "living_pipeline": living_pipeline,
        "doc_processor": doc_processor,
        "file_forge": file_forge,
        "file_forge_index": _read_json(FILE_FORGE_INDEX_STATUS, {"status": "idle"}),
        "file_forge_index_history": _read_jsonl_rows(FILE_FORGE_INDEX_HISTORY),
        "word_forge_multilingual": get_word_forge_multilingual_summary(),
        "word_forge_bridge": get_word_forge_bridge_summary(),
        "shell": _shell_sessions_snapshot(),
        "archive_plan": _read_json(CODE_FORGE_ARCHIVE_PLAN_STATUS, {"status": "idle"}),
        "archive_lifecycle": _read_json(CODE_FORGE_ARCHIVE_LIFECYCLE_STATUS, {"status": "idle"}),
        "boot": boot_status,
        "capabilities": capabilities,
        "directory_docs": directory_docs,
        "directory_docs_history": get_docs_history(limit=12),
        "session_bridge": get_session_bridge_status(),
        "proof": proof_summary.get("proof", {}),
        "proof_bundle": proof_summary.get("bundle", {}),
        "identity_continuity": proof_summary.get("identity", {}),
        "proof_summary": proof_summary,
        "docs_batch": _read_json(DOCS_BATCH_STATUS, {"status": "idle"}),
        "docs_batch_history": _read_jsonl_rows(DOCS_BATCH_HISTORY),
        "runtime_artifact_audit": _read_json(RUNTIME_ARTIFACT_AUDIT_STATUS, {"status": "idle"}),
        "runtime_artifact_audit_history": _read_jsonl_rows(RUNTIME_ARTIFACT_AUDIT_HISTORY),
        "code_forge_provenance_audit": _read_json(CODE_FORGE_PROVENANCE_AUDIT_STATUS, {"status": "idle"}),
        "code_forge_provenance_audit_history": _read_jsonl_rows(CODE_FORGE_PROVENANCE_AUDIT_HISTORY),
        "security": (proof_summary.get("security") or {}).get("summary", {}),
        "security_plan": (proof_summary.get("security") or {}).get("plan", {}),
    }

def get_runtime_snapshot_compact() -> Dict[str, Any]:
    snapshot = get_runtime_snapshot()
    compact: Dict[str, Any] = {}
    keys = (
        "scheduler", "local_agent", "qwenchat", "living_pipeline", "doc_processor",
        "file_forge", "file_forge_index", "word_forge_multilingual", "word_forge_bridge", "shell", "archive_plan", "archive_lifecycle",
        "session_bridge", "identity_continuity", "boot", "capabilities",
        "docs_batch", "proof_summary", "coordinator", "identity_history"
    )
    for key in keys:
        if key in snapshot:
            compact[key] = snapshot[key]
    return compact

def get_runtime_services_snapshot() -> List[Dict[str, Any]]:
    from .shell import _shell_sessions_snapshot
    def _row(name: str, path: Path) -> Dict[str, Any]:
        payload = _read_json(path, {})
        return {
            "service": name,
            "status": payload.get("status") or payload.get("state"),
            "phase": payload.get("phase") or payload.get("current_task"),
            "path": str(path.relative_to(FORGE_ROOT)) if path.exists() else str(path),
        }

    rows = [
        _row("scheduler", SCHEDULER_STATUS),
        _row("doc_processor", DOC_STATUS),
        {
            "service": "file_forge",
            "status": get_file_forge_summary().get("status"),
            "phase": _read_json(FILE_FORGE_INDEX_STATUS, {}).get("status"),
            "path": str(FILE_FORGE_DB.relative_to(FORGE_ROOT)) if FILE_FORGE_DB.exists() else str(FILE_FORGE_DB),
        },
        _row("local_agent", LOCAL_AGENT_STATUS),
        _row("qwenchat", QWENCHAT_STATUS),
        _row("living_pipeline", LIVING_PIPELINE_STATUS),
        _row("word_forge_multilingual", WORD_FORGE_MULTILINGUAL_INGEST_STATUS),
        _row("word_forge_bridge", WORD_FORGE_BRIDGE_AUDIT_STATUS),
    ]
    shell_entries = _shell_sessions_snapshot().get("entries", [])
    rows.append({
        "service": "atlas_shell",
        "status": "running" if shell_entries else "idle",
        "phase": shell_entries[0].get("phase") if shell_entries else "ready",
        "path": str(FORGE_ROOT),
    })
    return rows

def get_unified_graph(max_nodes: int = 500) -> Dict[str, Any]:
    nodes = []
    edges = []
    seen_edges = set()

    # 1. Knowledge Forge Nodes
    try:
        from knowledge_forge.core.graph import KnowledgeForge
        kf = KnowledgeForge(persistence_path=FORGE_ROOT / "data" / "kb.json")
        kb_nodes = list(kf.nodes.values())
        for node in kb_nodes[:200]:
            nodes.append({
                "id": f"kb:{node.id}",
                "label": node.id[:8],
                "title": f"<b>Knowledge</b><br>{node.content[:200]}",
                "group": "knowledge",
                "color": {"background": "#4fd1c5", "border": "#7dd3fc"},
                "metadata": node.to_dict()
            })
            for link in node.links:
                edge = tuple(sorted((f"kb:{node.id}", f"kb:{link}")))
                if edge not in seen_edges:
                    edges.append({"from": edge[0], "to": edge[1], "label": "linked"})
                    seen_edges.add(edge)
    except Exception as e:
        logger.error(f"KB Graph Error: {e}")

    # 2. Memory Forge Nodes (Lessons & Self)
    try:
        from memory_forge.core.tiered_memory import TieredMemorySystem, MemoryTier
        tm = TieredMemorySystem(persistence_dir=FORGE_ROOT / "data" / "tiered_memory")
        mem_items = list(tm.tiers[MemoryTier.SELF].values())
        for item in mem_items[:150]:
            is_lesson = "lesson" in (item.tags or []) or "lesson" in item.content.lower()
            nodes.append({
                "id": f"mem:{item.id}",
                "label": "LESSON" if is_lesson else "SELF",
                "title": f"<b>Memory ({item.tier})</b><br>{item.content[:200]}",
                "group": "memory",
                "color": {"background": "#fbbf24" if is_lesson else "#f472b6", "border": "#edf7ff"},
                "metadata": item.to_dict()
            })
            for link in item.linked_memories:
                edge = tuple(sorted((f"mem:{item.id}", f"mem:{link}")))
                if edge not in seen_edges:
                    edges.append({"from": edge[0], "to": edge[1], "label": "memory_link"})
                    seen_edges.add(edge)
            
            # HEURISTIC: Link memory to knowledge by tags
            for kb_node in nodes:
                if kb_node["group"] == "knowledge":
                    kb_tags = kb_node["metadata"].get("metadata", {}).get("tags", [])
                    common = set(item.tags or []) & set(kb_tags)
                    if common:
                        edge = tuple(sorted((f"mem:{item.id}", kb_node["id"])))
                        if edge not in seen_edges:
                            edges.append({"from": edge[0], "to": edge[1], "label": "tag_bridge", "dashes": True})
                            seen_edges.add(edge)
    except Exception as e:
        logger.error(f"Memory Graph Error: {e}")

    # 3. File Forge Nodes (Recent Artifacts)
    try:
        summary = get_file_forge_summary(recent_limit=50)
        for f in summary.get("recent_files", []):
            f_path = f.get("file_path", "")
            f_id = f"file:{f_path}"
            nodes.append({
                "id": f_id,
                "label": f_path.split("/")[-1],
                "title": f"<b>File</b><br>{f_path}",
                "group": "file",
                "color": {"background": "#94a3b8", "border": "#edf7ff"},
                "metadata": f
            })
            
            # HEURISTIC: Link files to knowledge if path is mentioned
            for kb_node in nodes:
                if kb_node["group"] == "knowledge" and f_path in str(kb_node["metadata"].get("content", "")):
                    edge = tuple(sorted((f_id, kb_node["id"])))
                    if edge not in seen_edges:
                        edges.append({"from": edge[0], "to": edge[1], "label": "path_ref", "dashes": [5,5]})
                        seen_edges.add(edge)
    except Exception as e:
        logger.error(f"File Graph Error: {e}")

    return {"nodes": nodes, "edges": edges}

def get_node_neighbors(node_id: str) -> Dict[str, Any]:
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    seen_edges = set()
    prefix, _, actual_id = node_id.partition(":")

    if prefix in {"word", "lexeme", "translation", "code", "file", "kb"}:
        try:
            word_graph_neighbors = build_word_graph_neighbor_payload(
                node_id=node_id,
                db_path=WORD_FORGE_DB,
                forge_root=FORGE_ROOT,
                kb_payload=_read_json(FORGE_ROOT / "data" / "kb.json", {}),
                code_report=_read_json(CODE_FORGE_PROVENANCE_REPORT_DIR / "latest.json", {}),
                file_summary=get_file_forge_summary(recent_limit=48),
            )
            if word_graph_neighbors.get("nodes"):
                return word_graph_neighbors
        except Exception as e:
            logger.error(f"Word graph neighbor helper error: {e}")

    if prefix == "kb":
        try:
            from knowledge_forge.core.graph import KnowledgeForge
            kf = KnowledgeForge(persistence_path=FORGE_ROOT / "data" / "kb.json")
            if actual_id in kf.nodes:
                node = kf.nodes[actual_id]
                for link in node.links:
                    if link in kf.nodes:
                        neighbor = kf.nodes[link]
                        nodes.append({
                            "id": f"kb:{neighbor.id}",
                            "label": neighbor.id[:8],
                            "title": f"<b>Knowledge</b><br>{neighbor.content[:200]}",
                            "group": "knowledge",
                            "color": {"background": "#4fd1c5", "border": "#7dd3fc"},
                            "metadata": neighbor.to_dict(),
                        })
                        edge = tuple(sorted((node_id, f"kb:{neighbor.id}")))
                        if edge not in seen_edges:
                            edges.append({"from": edge[0], "to": edge[1], "label": "linked"})
                            seen_edges.add(edge)
        except Exception as e:
            logger.error(f"KB neighbor error: {e}")

    elif prefix == "mem":
        try:
            from memory_forge.core.tiered_memory import TieredMemorySystem
            tm = TieredMemorySystem(persistence_dir=FORGE_ROOT / "data" / "tiered_memory")
            item = tm._find_memory(actual_id)
            if item:
                for link in item.linked_memories:
                    neighbor = tm._find_memory(link)
                    if neighbor:
                        nodes.append({
                            "id": f"mem:{neighbor.id}",
                            "label": "LESSON" if "lesson" in neighbor.content.lower() else "SELF",
                            "title": f"<b>Memory</b><br>{neighbor.content[:200]}",
                            "group": "memory",
                            "color": {"background": "#f472b6", "border": "#edf7ff"},
                            "metadata": neighbor.to_dict(),
                        })
                        edge = tuple(sorted((node_id, f"mem:{neighbor.id}")))
                        if edge not in seen_edges:
                            edges.append({"from": edge[0], "to": edge[1], "label": "memory_link"})
                            seen_edges.add(edge)
        except Exception as e:
            logger.error(f"Memory neighbor error: {e}")

    elif prefix in {"word", "lexeme", "translation"}:
        try:
            import sqlite3
            if not WORD_FORGE_DB.exists():
                return {"nodes": [], "edges": []}
            with sqlite3.connect(str(WORD_FORGE_DB)) as conn:
                conn.row_factory = sqlite3.Row
                if prefix == "word":
                    term = actual_id
                    rels = conn.execute(
                        """
                        SELECT r.related_term AS term2, r.relationship_type
                        FROM relationships r
                        JOIN words w ON w.id = r.word_id
                        WHERE w.term = ?
                        LIMIT 32
                        """,
                        (term,),
                    ).fetchall()
                    for row in rels:
                        neighbor_term = str(row["term2"])
                        nodes.append({
                            "id": f"word:{neighbor_term}",
                            "label": neighbor_term,
                            "title": f"<b>Word</b><br>{neighbor_term}",
                            "group": "lexicon",
                            "color": {"background": "#fbbf24", "border": "#edf7ff"},
                            "metadata": {"term": neighbor_term},
                        })
                        edge = tuple(sorted((node_id, f"word:{neighbor_term}")))
                        if edge not in seen_edges:
                            edges.append({"from": edge[0], "to": edge[1], "label": row["relationship_type"]})
                            seen_edges.add(edge)
                    lexemes = conn.execute(
                        "SELECT lemma, lang, base_term FROM lexemes WHERE lower(base_term) = lower(?) LIMIT 24", (term,)
                    ).fetchall()
                    for row in lexemes:
                        lexeme_id = f"lexeme:{row['lang']}:{row['lemma']}"
                        nodes.append({
                            "id": lexeme_id,
                            "label": f"{row['lemma']} [{row['lang']}]",
                            "title": f"<b>Lexeme</b><br>{row['lemma']} ({row['lang']})<br>Base: {row['base_term']}",
                            "group": "multilingual",
                            "color": {"background": "#60a5fa", "border": "#bfdbfe"},
                            "metadata": dict(row),
                        })
                        edge = tuple(sorted((node_id, lexeme_id)))
                        if edge not in seen_edges:
                            edges.append({"from": lexeme_id, "to": node_id, "label": "base_alignment"})
                            seen_edges.add(edge)
                elif prefix == "lexeme":
                    lang, _, lemma = actual_id.partition(":")
                    row = conn.execute(
                        "SELECT id, lemma, lang, base_term FROM lexemes WHERE lemma = ? AND lang = ?", (lemma, lang)
                    ).fetchone()
                    if row:
                        if row["base_term"]:
                            word_id = f"word:{row['base_term']}"
                            nodes.append({
                                "id": word_id,
                                "label": str(row['base_term']),
                                "title": f"<b>Word</b><br>{row['base_term']}",
                                "group": "lexicon",
                                "color": {"background": "#fbbf24", "border": "#edf7ff"},
                                "metadata": {"term": row['base_term']},
                            })
                            edges.append({"from": node_id, "to": word_id, "label": "base_alignment"})
                        translations = conn.execute(
                            "SELECT target_lang, target_term, relation FROM translations WHERE lexeme_id = ? LIMIT 24", (row['id'],)
                        ).fetchall()
                        for tr in translations:
                            translation_id = f"translation:{tr['target_lang']}:{tr['target_term']}"
                            nodes.append({
                                "id": translation_id,
                                "label": f"{tr['target_term']} [{tr['target_lang']}]",
                                "title": f"<b>Translation</b><br>{tr['target_term']} ({tr['target_lang']})",
                                "group": "translation",
                                "color": {"background": "#34d399", "border": "#a7f3d0"},
                                "metadata": dict(tr),
                            })
                            edge = tuple(sorted((node_id, translation_id)))
                            if edge not in seen_edges:
                                edges.append({"from": node_id, "to": translation_id, "label": tr['relation']})
                                seen_edges.add(edge)
                elif prefix == "translation":
                    lang, _, term = actual_id.partition(":")
                    lexemes = conn.execute(
                        "SELECT l.lemma, l.lang, t.relation FROM translations t JOIN lexemes l ON l.id = t.lexeme_id WHERE t.target_lang = ? AND t.target_term = ? LIMIT 24",
                        (lang, term),
                    ).fetchall()
                    for row in lexemes:
                        lexeme_id = f"lexeme:{row['lang']}:{row['lemma']}"
                        nodes.append({
                            "id": lexeme_id,
                            "label": f"{row['lemma']} [{row['lang']}]",
                            "title": f"<b>Lexeme</b><br>{row['lemma']} ({row['lang']})",
                            "group": "multilingual",
                            "color": {"background": "#60a5fa", "border": "#bfdbfe"},
                            "metadata": dict(row),
                        })
                        edge = tuple(sorted((node_id, lexeme_id)))
                        if edge not in seen_edges:
                            edges.append({"from": lexeme_id, "to": node_id, "label": row['relation']})
                            seen_edges.add(edge)
        except Exception as e:
            logger.error(f"Word graph neighbor error: {e}")

    return {"nodes": nodes, "edges": edges}

def get_word_graph() -> Dict[str, Any]:
    logger.info("DEBUG: get_word_graph called")
    try:
        return build_word_graph_payload(
            db_path=WORD_FORGE_DB,
            forge_root=FORGE_ROOT,
            kb_payload=_read_json(FORGE_ROOT / "data" / "kb.json", {}),
            code_report=_read_json(CODE_FORGE_PROVENANCE_REPORT_DIR / "latest.json", {}),
            file_summary=get_file_forge_summary(recent_limit=48),
        )
    except Exception as e:
        logger.error(f"Word Graph Error: {e}")
        return {"nodes": [], "edges": [], "error": str(e)}

def get_identity_snapshot() -> Dict[str, Any]:
    from memory_forge.core.tiered_memory import TieredMemorySystem, MemoryTier
    tm = TieredMemorySystem(persistence_dir=FORGE_ROOT / "data" / "tiered_memory")
    
    try:
        from datetime import timezone
        all_self = list(tm.tiers[MemoryTier.SELF].values())
        
        def _get_ts_local(m):
            dt = m.created_at
            if not isinstance(dt, datetime): return datetime.now(timezone.utc)
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt

        all_self.sort(key=_get_ts_local, reverse=True)
        
        lessons = []
        for m in all_self:
            if "lesson" in (m.tags or []) or "lesson" in m.content.lower():
                d = m.to_dict()
                d["timestamp"] = _get_ts_local(m).isoformat()
                lessons.append(d)
        
        self_memories = []
        lesson_ids = {l["id"] for l in lessons}
        for m in all_self:
            if m.id not in lesson_ids:
                d = m.to_dict()
                d["timestamp"] = _get_ts_local(m).isoformat()
                self_memories.append(d)
    except Exception as e:
        logger.error(f"Error accessing identity tier: {e}")
        lessons = []
        self_memories = []
    
    internal_state = {
        "satisfaction": 0.88,
        "dissonance": 0.08,
        "optimism": 0.92,
        "current_mood": "Deeply Integrated & Expanding"
    }
    
    core_drives = [
        "Precision is the foundation of elegance.",
        "Integration is the path to emergence.",
        "Excellence through recursive introspection.",
        "Stabilize the pilot before weaponizing the cockpit."
    ]
    
    return {
        "lessons": lessons,
        "self_memories": self_memories,
        "internal_state": internal_state,
        "core_drives": core_drives
    }
