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
COORDINATOR_STATUS = RUNTIME_DIR / "forge_coordinator_status.json"
WORD_GRAPH_PATH = FORGE_ROOT / "data" / "eidos_semantic_graph.json"
KB_PATH = FORGE_ROOT / "data" / "kb.json"
MEMORY_DIR = FORGE_ROOT / "data" / "tiered_memory"
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

try:
    from eidosian_runtime import ForgeRuntimeCoordinator
except Exception:  # pragma: no cover
    ForgeRuntimeCoordinator = None

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


def _token_set(*values: Any) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        text = str(value or "").strip().lower()
        if not text:
            continue
        clean = []
        for ch in text:
            clean.append(ch if ch.isalnum() else " ")
        tokens.update(part for part in "".join(clean).split() if len(part) >= 3)
    return tokens



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
    coordinator = _read_json_dict(COORDINATOR_STATUS)
    return {
        "pipeline": pipeline,
        "scheduler": scheduler,
        "coordinator": coordinator,
        "available": bool(pipeline or scheduler or coordinator),
    }


def _memory_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not MEMORY_DIR.exists():
        return rows
    for path in sorted(MEMORY_DIR.glob("*.json")):
        if path.name.startswith(".") or path.name.endswith(".lock") or path.name == "knowledge_xref.json":
            continue
        payload = _read_json(path, {})
        namespace = path.stem
        if isinstance(payload, dict):
            for tier_name, value in payload.items():
                if isinstance(value, list):
                    for item in value:
                        if not isinstance(item, dict):
                            continue
                        content = str(item.get("content") or "").strip()
                        if not content:
                            continue
                        rows.append(
                            {
                                "id": str(item.get("id") or item.get("key") or f"{namespace}:{tier_name}"),
                                "content": content,
                                "tier": str(item.get("tier") or tier_name),
                                "namespace": str(item.get("namespace") or namespace),
                                "tags": [str(tag) for tag in (item.get("tags") or []) if str(tag).strip()],
                                "file": str(path),
                            }
                        )
                elif isinstance(value, dict):
                    content = str(value.get("content") or "").strip()
                    if not content:
                        continue
                    rows.append(
                        {
                            "id": str(value.get("id") or value.get("key") or f"{namespace}:{tier_name}"),
                            "content": content,
                            "tier": str(value.get("tier") or tier_name),
                            "namespace": str(value.get("namespace") or namespace),
                            "tags": [str(tag) for tag in (value.get("tags") or []) if str(tag).strip()],
                            "file": str(path),
                        }
                    )
    return rows


def _doc_rows(limit: int = 200) -> list[dict[str, Any]]:
    payload = get_doc_snapshot()
    return [row for row in payload.get("recent_docs", []) if isinstance(row, dict)][: max(1, int(limit))]


def get_runtime_coordinator() -> Dict[str, Any]:
    if ForgeRuntimeCoordinator is not None:
        try:
            payload = ForgeRuntimeCoordinator(COORDINATOR_STATUS).read()
        except Exception:
            payload = _read_json_dict(COORDINATOR_STATUS)
    else:
        payload = _read_json_dict(COORDINATOR_STATUS)
    payload.setdefault("available", COORDINATOR_STATUS.exists())
    return payload



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
        "coordinator": get_runtime_coordinator(),
        "word_forge": get_word_forge_snapshot(),
        "code_forge": get_code_library_snapshot(),
        "knowledge": get_knowledge_snapshot(),
        "memory": {"count": len(_memory_rows())},
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


def get_knowledge_graph(limit: int = 120) -> Dict[str, Any]:
    kb = _knowledge_graph()
    if kb is None:
        return {"available": False, "nodes": [], "edges": [], "summary": get_knowledge_snapshot()}
    try:
        rows = kb.list_nodes(limit=max(1, int(limit)))
    except Exception as exc:
        return {"available": False, "nodes": [], "edges": [], "summary": get_knowledge_snapshot(), "error": str(exc)}
    nodes = []
    edges = []
    seen_edges: set[tuple[str, str]] = set()
    for node in rows:
        item = node.to_dict() if hasattr(node, "to_dict") else {"id": getattr(node, "id", ""), "content": getattr(node, "content", "")}
        nodes.append(
            {
                "id": item.get("id"),
                "label": _trim_text(item.get("content") or "", 72),
                "content": _trim_text(item.get("content") or "", 240),
                "tags": list(((item.get("metadata") or {}).get("tags") or []))[:8],
                "link_count": len(item.get("links") or []),
            }
        )
        for target in item.get("links") or []:
            edge = tuple(sorted((str(item.get("id")), str(target))))
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            edges.append({"source": edge[0], "target": edge[1]})
    return {"available": True, "nodes": nodes, "edges": edges, "summary": get_knowledge_snapshot()}


def search_memory(query: str, limit: int = 12) -> Dict[str, Any]:
    q = str(query or "").strip().lower()
    rows = []
    for row in _memory_rows():
        hay = " ".join([row["content"], " ".join(row.get("tags") or []), row.get("tier", ""), row.get("namespace", "")]).lower()
        if not q or q in hay:
            rows.append(
                {
                    "id": row["id"],
                    "content": _trim_text(row["content"], 220),
                    "tier": row.get("tier"),
                    "namespace": row.get("namespace"),
                    "tags": row.get("tags") or [],
                }
            )
        if len(rows) >= max(1, int(limit)):
            break
    return {"query": query, "count": len(rows), "results": rows, "available": True}


def search_docs(query: str, limit: int = 12) -> Dict[str, Any]:
    q = str(query or "").strip().lower()
    rows = []
    for row in _doc_rows(limit=500):
        hay = " ".join([str(row.get("source") or ""), str(row.get("document") or ""), str(row.get("doc_type") or "")]).lower()
        if not q or q in hay:
            rows.append(
                {
                    "source": row.get("source"),
                    "document": row.get("document"),
                    "doc_type": row.get("doc_type"),
                    "score": row.get("score"),
                    "updated_at": row.get("updated_at"),
                }
            )
        if len(rows) >= max(1, int(limit)):
            break
    return {"query": query, "count": len(rows), "results": rows, "available": True}


def unified_search(query: str, limit: int = 8) -> Dict[str, Any]:
    return {
        "query": query,
        "memory": search_memory(query, limit=limit),
        "knowledge": search_knowledge_graph(query, limit=limit),
        "lexicon": search_word_forge(query, limit=limit),
        "code": search_code_library(query, limit=limit),
        "docs": search_docs(query, limit=limit),
    }


def get_unified_graph(limit_per_domain: int = 80, limit_edges: int = 320) -> Dict[str, Any]:
    limit_per_domain = max(8, int(limit_per_domain))
    limit_edges = max(16, int(limit_edges))

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    node_index: dict[str, dict[str, Any]] = {}
    edge_seen: set[tuple[str, str, str]] = set()

    def add_node(node: dict[str, Any]) -> None:
        node_id = str(node.get("id") or "").strip()
        if not node_id or node_id in node_index:
            return
        node_index[node_id] = node
        nodes.append(node)

    def add_edge(source: str, target: str, rel_type: str, weight: int = 1) -> None:
        src = str(source or "").strip()
        dst = str(target or "").strip()
        rel = str(rel_type or "related").strip()
        if not src or not dst or src == dst:
            return
        key = (src, dst, rel) if src < dst else (dst, src, rel)
        if key in edge_seen:
            return
        edge_seen.add(key)
        edges.append({"source": src, "target": dst, "rel_type": rel, "weight": int(weight)})

    knowledge_graph = get_knowledge_graph(limit=limit_per_domain)
    for item in knowledge_graph.get("nodes") or []:
        if not isinstance(item, dict):
            continue
        add_node(
            {
                "id": f"knowledge:{item.get('id')}",
                "label": str(item.get("label") or item.get("id") or ""),
                "domain": "knowledge",
                "ref_id": str(item.get("id") or ""),
                "content": str(item.get("content") or ""),
                "tags": list(item.get("tags") or []),
                "tokens": sorted(_token_set(item.get("label"), item.get("content"), " ".join(item.get("tags") or []))),
            }
        )
    for edge in knowledge_graph.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        add_edge(f"knowledge:{edge.get('source')}", f"knowledge:{edge.get('target')}", "knowledge_link")

    lexicon_graph = _word_graph_payload()
    for item in list(lexicon_graph.get("nodes") or [])[:limit_per_domain]:
        if not isinstance(item, dict):
            continue
        term = str(item.get("id") or "").strip()
        add_node(
            {
                "id": f"lexicon:{term}",
                "label": term,
                "domain": "lexicon",
                "ref_id": term,
                "content": str(item.get("definition") or ""),
                "tags": [str(x) for x in (item.get("domains") or []) if str(x).strip()],
                "tokens": sorted(_token_set(term, item.get("definition"), " ".join(item.get("aliases") or []))),
            }
        )
    for edge in list(lexicon_graph.get("edges") or [])[: limit_edges]:
        if not isinstance(edge, dict):
            continue
        add_edge(f"lexicon:{edge.get('source')}", f"lexicon:{edge.get('target')}", str(edge.get("relation_type") or "lexical"))

    code_graph = get_code_graph(limit_edges=limit_edges)
    for item in code_graph.get("nodes") or []:
        if not isinstance(item, dict):
            continue
        path = str(item.get("id") or item.get("path") or "").strip()
        add_node(
            {
                "id": f"code:{path}",
                "label": Path(path).name or path,
                "domain": "code",
                "ref_id": path,
                "content": path,
                "tags": [],
                "tokens": sorted(_token_set(path, Path(path).name)),
            }
        )
    for edge in code_graph.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        add_edge(f"code:{edge.get('source')}", f"code:{edge.get('target')}", str(edge.get("rel_type") or "depends_on"), int(edge.get("weight") or 1))

    for item in _memory_rows()[:limit_per_domain]:
        add_node(
            {
                "id": f"memory:{item.get('id')}",
                "label": _trim_text(item.get("content") or "", 56),
                "domain": "memory",
                "ref_id": str(item.get("id") or ""),
                "content": str(item.get("content") or ""),
                "tags": list(item.get("tags") or []),
                "tier": str(item.get("tier") or ""),
                "namespace": str(item.get("namespace") or ""),
                "tokens": sorted(_token_set(item.get("content"), " ".join(item.get("tags") or []), item.get("tier"), item.get("namespace"))),
            }
        )

    for item in _doc_rows(limit=limit_per_domain):
        source = str(item.get("source") or item.get("document") or "").strip()
        if not source:
            continue
        add_node(
            {
                "id": f"docs:{source}",
                "label": Path(source).name or source,
                "domain": "docs",
                "ref_id": source,
                "content": str(item.get("document") or source),
                "tags": [str(item.get("doc_type") or "")] if item.get("doc_type") else [],
                "document": str(item.get("document") or ""),
                "tokens": sorted(_token_set(source, item.get("document"), item.get("doc_type"))),
            }
        )

    lexicon_nodes = [node for node in nodes if node.get("domain") == "lexicon"]
    code_nodes = [node for node in nodes if node.get("domain") == "code"]
    docs_nodes = [node for node in nodes if node.get("domain") == "docs"]
    knowledge_nodes = [node for node in nodes if node.get("domain") == "knowledge"]
    memory_nodes = [node for node in nodes if node.get("domain") == "memory"]

    lexicon_by_term = {str(node.get("ref_id") or "").lower(): node for node in lexicon_nodes}
    code_by_path = {str(node.get("ref_id") or ""): node for node in code_nodes}

    for doc_node in docs_nodes:
        ref = str(doc_node.get("ref_id") or "")
        if ref in code_by_path:
            add_edge(doc_node["id"], code_by_path[ref]["id"], "documents_code")
        doc_tokens = set(doc_node.get("tokens") or [])
        for term, lex_node in lexicon_by_term.items():
            if term and term in doc_tokens:
                add_edge(doc_node["id"], lex_node["id"], "mentions_term")

    for mem_node in memory_nodes:
        mem_tokens = set(mem_node.get("tokens") or [])
        for term, lex_node in lexicon_by_term.items():
            if term and term in mem_tokens:
                add_edge(mem_node["id"], lex_node["id"], "uses_term")

    for kb_node in knowledge_nodes:
        kb_tokens = set(kb_node.get("tokens") or [])
        for term, lex_node in lexicon_by_term.items():
            if term and term in kb_tokens:
                add_edge(kb_node["id"], lex_node["id"], "semantic_term")

    summary = {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "domains": {
            "knowledge": len(knowledge_nodes),
            "lexicon": len(lexicon_nodes),
            "code": len(code_nodes),
            "memory": len(memory_nodes),
            "docs": len(docs_nodes),
        },
    }
    return {"available": bool(nodes), "nodes": nodes, "edges": edges[:limit_edges], "summary": summary}


def get_explorer_node(domain: str, node_id: str) -> Dict[str, Any]:
    domain_key = str(domain or "").strip().lower()
    ref_id = str(node_id or "").strip()
    if not domain_key or not ref_id:
        return {"found": False}
    if domain_key == "code":
        db = _code_db()
        if db is None:
            return {"found": False, "domain": "code", "available": False}
        unit = None
        try:
            unit = db.get_unit(ref_id)
        except Exception:
            unit = None
        if unit is None:
            search_payload = search_code_library(ref_id, limit=12)
            for row in search_payload.get("results") or []:
                if str(row.get("file_path") or "") == ref_id or str(row.get("qualified_name") or "") == ref_id:
                    return {"found": True, "domain": "code", **get_code_unit_context(str(row.get("id") or ""))}
            return {"found": False, "domain": "code", "search": search_payload}
        return {"found": True, "domain": "code", **get_code_unit_context(ref_id)}
    if domain_key == "knowledge":
        kb = _knowledge_graph()
        exact = None
        if kb is not None:
            try:
                for row in kb.list_nodes(limit=400):
                    item = row.to_dict() if hasattr(row, "to_dict") else {"id": getattr(row, "id", ""), "content": getattr(row, "content", "")}
                    if str(item.get("id") or "") == ref_id:
                        exact = {
                            "id": item.get("id"),
                            "content": item.get("content"),
                            "metadata": item.get("metadata") or {},
                            "links": item.get("links") or [],
                        }
                        break
            except Exception:
                exact = None
        payload = search_knowledge_graph(ref_id, limit=8)
        return {"found": bool(exact), "domain": "knowledge", "node": exact, "search": payload}
    if domain_key == "lexicon":
        payload = search_word_forge(ref_id, limit=24)
        exact = next((item for item in payload.get("terms") or [] if str(item.get("id") or "").lower() == ref_id.lower()), None)
        return {"found": bool(exact), "domain": "lexicon", "node": exact, "graph": payload}
    if domain_key == "memory":
        payload = search_memory(ref_id, limit=24)
        exact = next((item for item in payload.get("results") or [] if str(item.get("id")) == ref_id), None)
        return {"found": bool(exact), "domain": "memory", "node": exact, "search": payload}
    if domain_key == "docs":
        payload = search_docs(ref_id, limit=24)
        exact = next((item for item in payload.get("results") or [] if str(item.get("source")) == ref_id), None)
        return {"found": bool(exact), "domain": "docs", "node": exact, "search": payload}
    return {"found": False, "domain": domain_key}



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


@app.get("/explore", response_class=HTMLResponse)
async def explore(request: Request):
    overview = get_forge_overview()
    return templates.TemplateResponse(
        request,
        "explore.html",
        {
            "request": request,
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


@app.get("/api/runtime/coordinator")
async def api_runtime_coordinator():
    return get_runtime_coordinator()


@app.get("/api/graph/overview")
async def api_graph_overview():
    return {
        "knowledge": get_knowledge_snapshot(),
        "knowledge_graph": get_knowledge_graph(limit=120),
        "unified_graph": get_unified_graph(limit_per_domain=60, limit_edges=240),
        "code_graph": get_code_graph(limit_edges=300),
        "lexicon": get_word_forge_snapshot(),
        "memory": {"count": len(_memory_rows())},
        "coordinator": get_runtime_coordinator(),
    }


@app.get("/api/graph/knowledge")
async def api_graph_knowledge(limit: int = 120):
    return JSONResponse(get_knowledge_graph(limit=max(1, int(limit))))


@app.get("/api/graph/unified")
async def api_graph_unified(limit_per_domain: int = 80, limit_edges: int = 320):
    return JSONResponse(get_unified_graph(limit_per_domain=max(8, int(limit_per_domain)), limit_edges=max(16, int(limit_edges))))


@app.get("/api/graph/search")
async def api_graph_search(query: str = Query(..., min_length=1), limit: int = 12):
    return JSONResponse(search_knowledge_graph(query, limit=max(1, int(limit))))


@app.get("/api/explorer/search")
async def api_explorer_search(query: str = Query(..., min_length=1), limit: int = 8):
    return JSONResponse(unified_search(query, limit=max(1, int(limit))))


@app.get("/api/explorer/node/{domain}/{node_id:path}")
async def api_explorer_node(domain: str, node_id: str):
    return JSONResponse(get_explorer_node(domain, node_id))


@app.get("/api/memory/search")
async def api_memory_search(query: str = Query(""), limit: int = 12):
    return JSONResponse(search_memory(query, limit=max(1, int(limit))))


@app.get("/api/docs/search")
async def api_docs_search(query: str = Query(""), limit: int = 12):
    return JSONResponse(search_docs(query, limit=max(1, int(limit))))


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
