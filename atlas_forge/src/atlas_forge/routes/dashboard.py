import os
import logging
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from ..system import get_system_stats
from ..forge import (
    get_forge_status, get_doc_snapshot, 
    get_docs_history, get_runtime_snapshot_compact, 
    get_runtime_services_snapshot, get_proof_summary,
    get_code_forge_provenance_audit_history, get_local_agent_history,
    get_scheduler_history, get_doc_processor_history, 
    get_qwenchat_history, get_living_pipeline_history,
    get_identity_snapshot, get_word_forge_multilingual_summary,
    get_word_forge_fasttext_summary, get_word_forge_bridge_summary, get_word_forge_multilingual_history,
    get_word_forge_fasttext_history, get_word_forge_bridge_history
)
from ..utils import _detect_lan_ip
from ..jobs import _service_command
from ..templating import templates

router = APIRouter()

logger = logging.getLogger("eidos_dashboard")

def _atlas_access_snapshot(request: Request) -> dict:
    port = request.url.port or int(os.environ.get("EIDOS_DASHBOARD_PORT", 8936))
    lan_ip = _detect_lan_ip()
    return {
        "port": port,
        "browse_localhost": f"http://localhost:{port}/browse/",
        "browse_loopback": f"http://127.0.0.1:{port}/browse/",
        "browse_lan": f"http://{lan_ip}:{port}/browse/" if lan_ip else "",
        "lan_ip": lan_ip or "",
    }

@router.get("/identity", response_class=HTMLResponse)
async def identity_page(request: Request):
    snapshot = get_identity_snapshot()
    return templates.TemplateResponse("identity.html", {"request": request, "identity": snapshot})

@router.get("/graphs", response_class=HTMLResponse)
async def graphs_page(request: Request):
    return templates.TemplateResponse(request, "graphs.html", {})

@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    try:
        sys_stats = get_system_stats()
        forge_status = get_forge_status()
        doc_snapshot = get_doc_snapshot()
        runtime_snapshot = get_runtime_snapshot_compact()
        runtime_services = get_runtime_services_snapshot()
        atlas_access = _atlas_access_snapshot(request)
        proof_summary = get_proof_summary()
        
        # Ensure 'runtime_snapshot' is never None or undefined for the template
        if runtime_snapshot is None:
            runtime_snapshot = {}

        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "request": request,
                "sys_stats": sys_stats,
                "forge_status": forge_status,
                "recent_docs": doc_snapshot.get("recent_docs", []),
                "doc_status": doc_snapshot.get("status", {}),
                "doc_index_count": doc_snapshot.get("index_count", 0),
                "docs_history": get_docs_history(limit=24),
                "runtime_snapshot": runtime_snapshot,
                "runtime_services": runtime_services,
                "code_forge_provenance_audit_history": get_code_forge_provenance_audit_history(),
                "local_agent_history": get_local_agent_history(),
                "scheduler_history": get_scheduler_history(),
                "doc_processor_history": get_doc_processor_history(),
                "qwenchat_history": get_qwenchat_history(),
                "living_pipeline_history": get_living_pipeline_history(),
                "word_forge_multilingual": get_word_forge_multilingual_summary(),
                "word_forge_fasttext": get_word_forge_fasttext_summary(),
                "word_forge_bridge": get_word_forge_bridge_summary(),
                "word_forge_multilingual_history": get_word_forge_multilingual_history(),
                "word_forge_fasttext_history": get_word_forge_fasttext_history(),
                "word_forge_bridge_history": get_word_forge_bridge_history(),
                "proof_snapshot": proof_summary.get("proof", {}),
                "proof_summary": proof_summary,
                "service_snapshot": await _service_command("status"),
                "atlas_access": atlas_access,
                # Placeholders for inventory if needed (though templates check existence)
                "docs_inventory": {}, 
                "docs_reviews": {},
                "docs_suppressed": {},
                "docs_tree": {"nodes": []},
            },
        )
    except Exception as e:
        logger.exception("Error rendering dashboard")
        return HTMLResponse(content=f"<h1>500 Internal Server Error</h1><pre>{e}</pre>", status_code=500)
