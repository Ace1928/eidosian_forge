from fastapi import APIRouter, Request, HTTPException
from ..system import get_system_stats
from ..forge import (
    get_doc_snapshot, get_runtime_services_snapshot, get_proof_summary,
    get_unified_graph, get_word_graph, get_file_forge_summary,
    get_node_neighbors
)
from ..jobs import _service_command, _scheduler_command
from ..shell import (
    _spawn_shell_session, _shell_session_payload, _stop_shell_session,
    SHELL_SESSIONS, SHELL_SESSIONS_LOCK
)
import os

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

@router.get("/runtime/file-forge")
async def api_file_forge_summary(path_prefix: str = "", recent_limit: int = 8):
    return get_file_forge_summary(path_prefix=path_prefix, recent_limit=recent_limit)

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
