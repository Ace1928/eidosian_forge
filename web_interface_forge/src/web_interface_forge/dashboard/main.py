import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List

import markdown
import psutil
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# --- Configuration ---
FORGE_ROOT = Path(os.environ.get("EIDOS_FORGE_ROOT", "/data/data/com.termux/files/home/eidosian_forge")).resolve()
DOC_RUNTIME = FORGE_ROOT / "doc_forge" / "runtime"
DOC_FINAL = DOC_RUNTIME / "final_docs"
DOC_INDEX = DOC_RUNTIME / "doc_index.json"
DOC_STATUS = DOC_RUNTIME / "processor_status.json"
TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eidos_dashboard")

# --- App Setup ---
app = FastAPI(title="Eidosian Atlas", version="1.0.0")
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
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        return {}


def _read_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        return default
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else default
    except Exception:
        return default

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

def get_file_tree(path: Path) -> List[Dict[str, Any]]:
    tree = []
    try:
        # Sort directories first, then files
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
                item["children"] = [] # Lazy loading not implemented for simplicity, just show dir
            tree.append(item)
    except Exception as e:
        logger.error(f"Error listing {path}: {e}")
    return tree

# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    sys_stats = get_system_stats()
    forge_status = get_forge_status()
    doc_snapshot = get_doc_snapshot()

    return templates.TemplateResponse(request, "index.html", {
        "request": request,
        "sys_stats": sys_stats,
        "forge_status": forge_status,
        "recent_docs": doc_snapshot["recent_docs"],
        "doc_status": doc_snapshot["status"],
        "doc_index_count": doc_snapshot["index_count"],
    })

@app.get("/browse/{path:path}", response_class=HTMLResponse)
async def browse(request: Request, path: str):
    # Security check: prevent directory traversal
    target_path = (DOC_FINAL / path).resolve()
    if not str(target_path).startswith(str(DOC_FINAL.resolve())):
        raise HTTPException(status_code=403, detail="Access denied")

    if not target_path.exists():
        raise HTTPException(status_code=404, detail="Path not found")

    if target_path.is_dir():
        files = get_file_tree(target_path)
        parent = str(Path(path).parent) if path != "." else None
        return templates.TemplateResponse(request, "browser.html", {
            "request": request,
            "path": path,
            "files": files,
            "parent": parent
        })
    elif target_path.is_file():
        if target_path.suffix.lower() == ".md":
            content = target_path.read_text()
            html_content = markdown.markdown(content, extensions=['fenced_code', 'tables', 'toc'])
            return templates.TemplateResponse(request, "viewer.html", {
                "request": request,
                "path": path,
                "content": html_content,
                "filename": target_path.name
            })
        else:
            # For non-markdown files, just serve raw or download
            return FileResponse(target_path)

@app.get("/api/system")
async def api_system():
    return get_system_stats()


@app.get("/api/doc/status")
async def api_doc_status():
    return get_doc_snapshot()


@app.get("/health")
async def health():
    return {"status": "ok", "service": "eidos_atlas"}

if __name__ == "__main__":
    import uvicorn
    # Default port 8936 for dashboard (next to doc_forge 8930)
    port = int(os.environ.get("EIDOS_DASHBOARD_PORT", 8936))
    uvicorn.run(app, host="0.0.0.0", port=port)
