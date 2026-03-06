from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import markdown
import psutil
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# --- Configuration ---
FORGE_ROOT = Path(os.environ.get("EIDOS_FORGE_ROOT", "/data/data/com.termux/files/home/eidosian_forge")).resolve()
for extra in (
    FORGE_ROOT / "lib",
    FORGE_ROOT / "code_forge" / "src",
    FORGE_ROOT / "knowledge_forge" / "src",
    FORGE_ROOT / "memory_forge" / "src",
    FORGE_ROOT / "eidos_mcp" / "src",
    FORGE_ROOT / "web_interface_forge" / "src",
    FORGE_ROOT,
):
    extra_str = str(extra)
    if extra.exists() and extra_str not in sys.path:
        sys.path.insert(0, extra_str)

DOC_RUNTIME = FORGE_ROOT / "doc_forge" / "runtime"
DOC_FINAL = DOC_RUNTIME / "final_docs"
DOC_INDEX = DOC_RUNTIME / "doc_index.json"
DOC_STATUS = DOC_RUNTIME / "processor_status.json"
RUNTIME_DIR = FORGE_ROOT / "data" / "runtime"
PIPELINE_STATUS = RUNTIME_DIR / "living_pipeline_status.json"
SCHEDULER_STATUS = RUNTIME_DIR / "eidos_scheduler_status.json"
WORD_GRAPH_PATH = FORGE_ROOT / "data" / "eidos_semantic_graph.json"
KB_PATH = FORGE_ROOT / "data" / "kb.json"
CODE_DB_PATH = FORGE_ROOT / "data" / "code_forge" / "library.sqlite"
GRAPHRAG_ROOT = (FORGE_ROOT / "graphrag_workspace") if (FORGE_ROOT / "graphrag_workspace").exists() else (FORGE_ROOT / "graphrag")
TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# --- Optional forge imports ---
try:
    from code_forge.library.db import CodeLibraryDB
except Exception:  # pragma: no cover
    CodeLibraryDB = None

try:
    from eidos_mcp.embeddings import SimpleEmbedder
except Exception:  # pragma: no cover
    SimpleEmbedder = None

try:
    from knowledge_forge.core.graph import KnowledgeForge
    from knowledge_forge.integrations.graphrag import GraphRAGIntegration
except Exception:  # pragma: no cover
    KnowledgeForge = None
    GraphRAGIntegration = None

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eidos_dashboard")

# --- App Setup ---
app = FastAPI(title="Eidosian Atlas", version="2.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# --- Helpers ---
def get_system_stats() -> Dict[str, Any]:
    try:
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage(str(FORGE_ROOT))
        return {
            "cpu": cpu,
            "ram_percent": mem.percent,
            "ram_used_gb": round(mem.used / (1024**3), 2),
            "ram_total_gb": round(mem.total / (1024**3), 2),
            "disk_percent": disk.percent,
            "disk_free_gb": round(disk.free / (1024**3), 2),
            "uptime": int(psutil.boot_time()),
        }
    except Exception as exc:
        logger.error("Error getting system stats: %s", exc)
        return {}



def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload
    except Exception:
        return default



def _read_json_dict(path: Path) -> Dict[str, Any]:
    payload = _read_json(path, {})
    return payload if isinstance(payload, dict) else {}



def _trim_text(text: Any, limit: int = 240) -> str:
    out = str(text or "").strip()
    if len(out) <= limit:
        return out
    return out[: limit - 3].rstrip() + "..."



def _word_graph_payload() -> Dict[str, Any]:
    payload = _read_json_dict(WORD_GRAPH_PATH)
    nodes = payload.get("nodes") if isinstance(payload.get("nodes"), list) else []
    edges = payload.get("edges") if isinstance(payload.get("edges"), list) else []
    return {"nodes": nodes, "edges": edges}



def _code_db() -> Any:
    if CodeLibraryDB is None or not CODE_DB_PATH.exists():
        return None
    try:
        return CodeLibraryDB(CODE_DB_PATH)
    except Exception:
        return None



def _knowledge_graph() -> Any:
    if KnowledgeForge is None or not KB_PATH.exists():
        return None
    embedder = SimpleEmbedder() if SimpleEmbedder is not None else None
    try:
        return KnowledgeForge(persistence_path=KB_PATH, embedder=embedder)
    except Exception:
        return None



def _graphrag() -> Any:
    if GraphRAGIntegration is None or not GRAPHRAG_ROOT.exists():
        return None
    try:
        return GraphRAGIntegration(graphrag_root=GRAPHRAG_ROOT)
    except Exception:
        return None



def get_forge_status() -> Dict[str, Any]:
    status = {"doc_forge": "unknown", "details": {}, "scheduler": "unknown", "pipeline": "unknown"}
    if DOC_STATUS.exists():
        data = _read_json_dict(DOC_STATUS)
        status["doc_forge"] = data.get("status", "unknown")
        status["details"] = data
    pipeline = _read_json_dict(PIPELINE_STATUS)
    scheduler = _read_json_dict(SCHEDULER_STATUS)
    status["pipeline"] = str(pipeline.get("state") or "unknown")
    status["scheduler"] = str(scheduler.get("state") or "unknown")
    status["pipeline_phase"] = str(pipeline.get("phase") or "")
    return status



def get_doc_snapshot() -> Dict[str, Any]:
    status_payload = _read_json_dict(DOC_STATUS)
    index_payload = _read_json_dict(DOC_INDEX)
    entries = index_payload.get("entries", [])
    if not isinstance(entries, list):
        entries = []
    recent_docs = [entry for entry in entries if isinstance(entry, dict)]
    recent_docs.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return {
        "status": status_payload,
        "index_count": len(entries),
        "recent_docs": recent_docs[:12],
    }



def get_pipeline_snapshot() -> Dict[str, Any]:
    pipeline = _read_json_dict(PIPELINE_STATUS)
    scheduler = _read_json_dict(SCHEDULER_STATUS)
    return {
        "pipeline": pipeline,
        "scheduler": scheduler,
        "available": bool(pipeline or scheduler),
    }



def get_word_forge_snapshot() -> Dict[str, Any]:
    payload = _word_graph_payload()
    nodes = payload["nodes"]
    edges = payload["edges"]
    sample_terms = []
    for node in nodes[:10]:
        if not isinstance(node, dict):
            continue
        sample_terms.append(
            {
                "term": str(node.get("id") or ""),
                "definition": _trim_text(node.get("definition") or "", 120),
                "pos": str(node.get("pos") or ""),
            }
        )
    return {
        "available": WORD_GRAPH_PATH.exists(),
        "path": str(WORD_GRAPH_PATH),
        "term_count": len(nodes),
        "edge_count": len(edges),
        "sample_terms": sample_terms,
    }



def get_code_library_snapshot() -> Dict[str, Any]:
    db = _code_db()
    if db is None:
        return {"available": False, "path": str(CODE_DB_PATH)}
    try:
        return {
            "available": True,
            "path": str(CODE_DB_PATH),
            "total_units": db.count_units(),
            "units_by_type": db.count_units_by_type(),
            "units_by_language": db.count_units_by_language(),
            "relationship_counts": db.relationship_counts(),
            "vector_index": db.vector_index_stats(),
        }
    except Exception as exc:
        return {"available": False, "path": str(CODE_DB_PATH), "error": str(exc)}



def get_knowledge_snapshot() -> Dict[str, Any]:
    kb = _knowledge_graph()
    grag = _graphrag()
    payload: Dict[str, Any] = {
        "available": bool(kb is not None),
        "path": str(KB_PATH),
        "node_count": 0,
        "concept_count": 0,
        "report_summary": {},
        "trend_summary": {},
        "assessment_summary": {},
    }
    if kb is not None:
        try:
            stats = kb.stats()
            payload["node_count"] = int(stats.get("node_count") or 0)
            payload["concept_count"] = int(stats.get("concept_count") or 0)
        except Exception as exc:
            payload["error"] = str(exc)
    if grag is not None:
        try:
            payload["report_summary"] = grag.native_report_summary(limit=8)
        except Exception:
            payload["report_summary"] = {}
        try:
            payload["trend_summary"] = grag.native_trend_summary(limit=8)
        except Exception:
            payload["trend_summary"] = {}
        try:
            payload["assessment_summary"] = grag.native_assessment_summary()
        except Exception:
            payload["assessment_summary"] = {}
    return payload



def get_forge_overview() -> Dict[str, Any]:
    return {
        "system": get_system_stats(),
        "forge_status": get_forge_status(),
        "documents": get_doc_snapshot(),
        "pipeline": get_pipeline_snapshot(),
        "word_forge": get_word_forge_snapshot(),
        "code_forge": get_code_library_snapshot(),
        "knowledge": get_knowledge_snapshot(),
    }



def search_word_forge(query: str, limit: int = 12) -> Dict[str, Any]:
    payload = _word_graph_payload()
    nodes = payload["nodes"]
    edges = payload["edges"]
    q = str(query or "").strip().lower()
    if not q:
        matches = [node for node in nodes if isinstance(node, dict)][: max(1, int(limit))]
    else:
        matches = []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            hay = " ".join(
                [
                    str(node.get("id") or ""),
                    str(node.get("definition") or ""),
                    " ".join(str(x) for x in (node.get("aliases") or [])),
                    " ".join(str(x) for x in (node.get("domains") or [])),
                ]
            ).lower()
            if q in hay:
                matches.append(node)
            if len(matches) >= max(1, int(limit)):
                break
    matched_terms = {str(node.get("id") or "") for node in matches if isinstance(node, dict)}
    related_edges = [
        edge for edge in edges if isinstance(edge, dict) and {str(edge.get("source") or ""), str(edge.get("target") or "")}.intersection(matched_terms)
    ][: max(1, int(limit)) * 4]
    return {
        "query": query,
        "count": len(matches),
        "terms": matches,
        "edges": related_edges,
        "graph": {
            "term_count": len(nodes),
            "edge_count": len(edges),
        },
    }



def search_code_library(query: str, limit: int = 12) -> Dict[str, Any]:
    db = _code_db()
    if db is None:
        return {"query": query, "count": 0, "results": [], "available": False}
    try:
        rows = db.semantic_search(query, limit=max(1, int(limit)), backend="hybrid", min_score=0.0)
    except Exception as exc:
        return {"query": query, "count": 0, "results": [], "available": False, "error": str(exc)}
    results = []
    for row in rows:
        results.append(
            {
                "id": row.get("id"),
                "name": row.get("name"),
                "qualified_name": row.get("qualified_name"),
                "unit_type": row.get("unit_type"),
                "language": row.get("language"),
                "file_path": row.get("file_path"),
                "semantic_score": row.get("semantic_score"),
                "vector_score": row.get("vector_score"),
                "search_preview": _trim_text(row.get("search_preview") or row.get("semantic_text") or "", 220),
            }
        )
    return {"query": query, "count": len(results), "results": results, "available": True}



def get_code_unit_context(unit_id: str) -> Dict[str, Any]:
    db = _code_db()
    if db is None:
        return {"found": False, "available": False}
    try:
        payload = db.unit_context(unit_id, context_lines=4, contains_limit=80, relationship_limit=40)
    except Exception as exc:
        return {"found": False, "available": False, "error": str(exc)}
    if not payload.get("found"):
        return payload
    payload["parents"] = list(payload.get("parents") or [])[:20]
    payload["children"] = list(payload.get("children") or [])[:20]
    return payload



def get_code_graph(limit_edges: int = 300) -> Dict[str, Any]:
    db = _code_db()
    if db is None:
        return {"available": False, "nodes": [], "edges": [], "summary": {}}
    try:
        return {"available": True, **db.module_dependency_graph(limit_edges=max(1, int(limit_edges)))}
    except Exception as exc:
        return {"available": False, "nodes": [], "edges": [], "summary": {}, "error": str(exc)}



def search_knowledge_graph(query: str, limit: int = 12) -> Dict[str, Any]:
    kb = _knowledge_graph()
    if kb is None:
        return {"query": query, "count": 0, "results": [], "available": False}
    try:
        rows = kb.semantic_search(query, limit=max(1, int(limit)))
    except Exception as exc:
        return {"query": query, "count": 0, "results": [], "available": False, "error": str(exc)}
    results = []
    for node in rows:
        item = node.to_dict() if hasattr(node, "to_dict") else {"id": getattr(node, "id", ""), "content": getattr(node, "content", "")}
        results.append(
            {
                "id": item.get("id"),
                "content": _trim_text(item.get("content") or "", 220),
                "tags": list(((item.get("metadata") or {}).get("tags") or []))[:8],
                "links": len(item.get("links") or []),
            }
        )
    return {
        "query": query,
        "count": len(results),
        "results": results,
        "available": True,
        "assessment": get_knowledge_snapshot().get("assessment_summary", {}),
    }



def get_file_tree(path: Path) -> List[Dict[str, Any]]:
    tree = []
    try:
        entries = sorted(list(path.iterdir()), key=lambda x: (not x.is_dir(), x.name.lower()))
        for entry in entries:
            if entry.name.startswith(".") or entry.name == "__pycache__":
                continue
            item = {
                "name": entry.name,
                "path": str(entry.relative_to(DOC_FINAL)),
                "is_dir": entry.is_dir(),
                "size": entry.stat().st_size if entry.is_file() else 0,
                "mtime": datetime.fromtimestamp(entry.stat().st_mtime).isoformat(),
            }
            if entry.is_dir():
                item["children"] = []
            tree.append(item)
    except Exception as exc:
        logger.error("Error listing %s: %s", path, exc)
    return tree


# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    overview = get_forge_overview()
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "request": request,
            "sys_stats": overview["system"],
            "forge_status": overview["forge_status"],
            "recent_docs": overview["documents"]["recent_docs"],
            "doc_status": overview["documents"]["status"],
            "doc_index_count": overview["documents"]["index_count"],
            "forge_overview": overview,
        },
    )


@app.get("/browse/{path:path}", response_class=HTMLResponse)
async def browse(request: Request, path: str):
    target_path = (DOC_FINAL / path).resolve()
    if not str(target_path).startswith(str(DOC_FINAL.resolve())):
        raise HTTPException(status_code=403, detail="Access denied")
    if not target_path.exists():
        raise HTTPException(status_code=404, detail="Path not found")
    if target_path.is_dir():
        files = get_file_tree(target_path)
        parent = str(Path(path).parent) if path != "." else None
        return templates.TemplateResponse(request, "browser.html", {"request": request, "path": path, "files": files, "parent": parent})
    if target_path.suffix.lower() == ".md":
        content = target_path.read_text(encoding="utf-8", errors="ignore")
        html_content = markdown.markdown(content, extensions=["fenced_code", "tables", "toc"])
        return templates.TemplateResponse(
            request,
            "viewer.html",
            {"request": request, "path": path, "content": html_content, "filename": target_path.name},
        )
    return FileResponse(target_path)


@app.get("/api/system")
async def api_system():
    return get_system_stats()


@app.get("/api/doc/status")
async def api_doc_status():
    return get_doc_snapshot()


@app.get("/api/runtime/forge")
async def api_runtime_forge():
    return get_forge_overview()


@app.get("/api/graph/overview")
async def api_graph_overview():
    return {
        "knowledge": get_knowledge_snapshot(),
        "code_graph": get_code_graph(limit_edges=300),
        "lexicon": get_word_forge_snapshot(),
    }


@app.get("/api/graph/search")
async def api_graph_search(query: str = Query(..., min_length=1), limit: int = 12):
    return JSONResponse(search_knowledge_graph(query, limit=max(1, int(limit))))


@app.get("/api/lexicon/search")
async def api_lexicon_search(query: str = Query(""), limit: int = 12):
    return JSONResponse(search_word_forge(query, limit=max(1, int(limit))))


@app.get("/api/lexicon/graph")
async def api_lexicon_graph(limit: int = 120):
    payload = _word_graph_payload()
    return {
        "available": WORD_GRAPH_PATH.exists(),
        "nodes": list(payload["nodes"])[: max(1, int(limit))],
        "edges": list(payload["edges"])[: max(1, int(limit)) * 4],
        "summary": get_word_forge_snapshot(),
    }


@app.get("/api/code/search")
async def api_code_search(query: str = Query(..., min_length=1), limit: int = 12):
    return JSONResponse(search_code_library(query, limit=max(1, int(limit))))


@app.get("/api/code/unit/{unit_id}")
async def api_code_unit(unit_id: str):
    return JSONResponse(get_code_unit_context(unit_id))


@app.get("/api/code/graph")
async def api_code_graph(limit_edges: int = 300):
    return JSONResponse(get_code_graph(limit_edges=max(1, int(limit_edges))))


@app.get("/health")
async def health():
    return {"status": "ok", "service": "eidos_atlas", "runtime": get_forge_status()}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("EIDOS_DASHBOARD_PORT", 8936))
    uvicorn.run(app, host="0.0.0.0", port=port)
