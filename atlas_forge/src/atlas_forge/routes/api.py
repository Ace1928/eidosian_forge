from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from ..system import get_system_stats
from ..forge import (
    get_doc_snapshot,
    get_runtime_services_snapshot,
    get_proof_summary,
    get_unified_graph,
    get_word_graph,
    get_file_forge_summary,
    get_node_neighbors,
    get_word_forge_multilingual_summary,
    get_word_forge_fasttext_summary,
    get_word_forge_polyglot_summary,
    get_word_forge_bridge_summary,
    get_word_forge_multilingual_history,
    get_word_forge_fasttext_history,
    get_word_forge_polyglot_history,
    get_word_forge_bridge_history,
    get_word_graph_communities,
    render_word_forge_metrics,
)
from ..jobs import _service_command, _scheduler_command
from ..shell import (
    _spawn_shell_session, _shell_session_payload, _stop_shell_session,
    SHELL_SESSIONS, SHELL_SESSIONS_LOCK,
)
from ..config import FORGE_ROOT, WORD_FORGE_DB
import os
from pathlib import Path

router = APIRouter(prefix="/api")


@router.get("/health")
async def api_health():
    return {"ok": True, "service": "atlas_forge", "status": "healthy"}


@router.get("/system")
async def api_system():
    return get_system_stats()


@router.get("/doc/status")
async def api_doc_status():
    return get_doc_snapshot()


@router.get("/services")
async def api_services():
    return get_runtime_services_snapshot()


@router.get("/runtime/services")
async def api_runtime_services():
    return {"contract": "eidos.runtime.services.v1", "entries": get_runtime_services_snapshot()}


@router.get("/runtime/file-forge")
async def api_file_forge_summary(path_prefix: str = "", recent_limit: int = 8):
    return get_file_forge_summary(path_prefix=path_prefix, recent_limit=recent_limit)


@router.get("/word-forge/multilingual")
async def api_word_forge_multilingual():
    return get_word_forge_multilingual_summary()


@router.get("/word-forge/multilingual/history")
async def api_word_forge_multilingual_history(limit: int = 12):
    return {"entries": get_word_forge_multilingual_history(limit=max(1, min(limit, 60)))}


@router.get("/word-forge/fasttext")
async def api_word_forge_fasttext():
    return get_word_forge_fasttext_summary()


@router.get("/word-forge/fasttext/history")
async def api_word_forge_fasttext_history(limit: int = 12):
    return {"entries": get_word_forge_fasttext_history(limit=max(1, min(limit, 60)))}


@router.get("/word-forge/polyglot")
async def api_word_forge_polyglot():
    return get_word_forge_polyglot_summary()


@router.get("/word-forge/polyglot/history")
async def api_word_forge_polyglot_history(limit: int = 12):
    return {"entries": get_word_forge_polyglot_history(limit=max(1, min(limit, 60)))}


@router.post("/word-forge/multilingual/run")
async def api_word_forge_multilingual_run(source_path: str, source_type: str, limit: int | None = None, force: bool = False):
    from word_forge.multilingual.runtime import run_multilingual_ingest

    path = Path(source_path).expanduser().resolve()
    if not path.exists():
        raise HTTPException(status_code=404, detail="source path not found")
    return run_multilingual_ingest(
        repo_root=FORGE_ROOT,
        source_path=path,
        source_type=source_type,
        db_path=WORD_FORGE_DB,
        limit=limit,
        force=force,
    )


@router.post("/word-forge/fasttext/run")
async def api_word_forge_fasttext_run(
    source_path: str,
    lang: str,
    limit: int | None = None,
    bootstrap_lang: str | None = None,
    top_k: int = 1,
    min_score: float = 0.55,
    apply: bool = False,
    force: bool = False,
):
    from word_forge.multilingual.fasttext_runtime import run_fasttext_ingest

    path = Path(source_path).expanduser().resolve()
    if not path.exists():
        raise HTTPException(status_code=404, detail="source path not found")
    return run_fasttext_ingest(
        repo_root=FORGE_ROOT,
        source_path=path,
        lang=lang,
        db_path=WORD_FORGE_DB,
        vector_db_path=FORGE_ROOT / "data" / "word_forge_fasttext.sqlite",
        limit=limit,
        bootstrap_lang=bootstrap_lang,
        top_k=top_k,
        min_score=min_score,
        apply=apply,
        force=force,
    )


@router.post("/word-forge/polyglot/run")
async def api_word_forge_polyglot_run(lang: str | None = None, limit: int | None = None, force: bool = False):
    from word_forge.multilingual.polyglot_runtime import run_polyglot_decomposition

    return run_polyglot_decomposition(
        repo_root=FORGE_ROOT,
        db_path=WORD_FORGE_DB,
        lang=lang,
        limit=limit,
        force=force,
    )


@router.get("/word-forge/bridge-audit")
async def api_word_forge_bridge_audit():
    return get_word_forge_bridge_summary()


@router.get("/word-forge/bridge-audit/history")
async def api_word_forge_bridge_audit_history(limit: int = 12):
    return {"entries": get_word_forge_bridge_history(limit=max(1, min(limit, 60)))}


@router.get("/metrics/word-forge", response_class=PlainTextResponse)
async def api_word_forge_metrics():
    return PlainTextResponse(render_word_forge_metrics(), media_type="text/plain; version=0.0.4; charset=utf-8")


@router.post("/word-forge/bridge-audit/run")
async def api_word_forge_bridge_audit_run():
    from word_forge.bridge.audit import run_bridge_audit

    return run_bridge_audit(repo_root=FORGE_ROOT, db_path=WORD_FORGE_DB)


@router.post("/service/command")
async def api_service_command(action: str, service: str = ""):
    return await _service_command(action, service)


@router.post("/scheduler/command")
async def api_scheduler_command(action: str):
    return _scheduler_command(action)


@router.get("/graph/knowledge")
async def api_knowledge_graph(max_nodes: int = 500):
    return get_unified_graph(max_nodes=max_nodes)


@router.get("/graph/word")
async def api_word_graph():
    return get_word_graph()


@router.get("/graph/word/communities")
async def api_word_graph_communities(limit: int = 12):
    return get_word_graph_communities(limit=max(1, min(limit, 60)))


@router.get("/graph/neighbors/{node_id:path}")
async def api_node_neighbors(node_id: str):
    return get_node_neighbors(node_id)


@router.post("/shell/start")
async def api_shell_start(cwd: str = "", cols: int = 120, rows: int = 28):
    return _spawn_shell_session(cwd=cwd, cols=cols, rows=rows)


@router.get("/shell/read")
async def api_shell_read(session_id: str, max_bytes: int = 16384):
    return _shell_session_payload(session_id, include_output=True, max_bytes=max_bytes)


@router.post("/shell/input")
async def api_shell_input(session_id: str, text: str):
    with SHELL_SESSIONS_LOCK:
        session = SHELL_SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="shell session not found")

    master_fd = session["fd"]
    try:
        os.write(master_fd, text.encode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"write failed: {e}")
    return {"status": "ok"}


@router.post("/shell/stop")
async def api_shell_stop(session_id: str):
    return _stop_shell_session(session_id)


@router.get("/proof/latest")
async def api_proof_latest():
    summary = get_proof_summary()
    return summary.get("proof", {})


@router.get("/proof/summary")
async def api_proof_summary():
    return get_proof_summary()
