import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import markdown
import psutil
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from eidosian_runtime import collect_runtime_capabilities

# --- Configuration ---
FORGE_ROOT = Path(os.environ.get("EIDOS_FORGE_ROOT", "/data/data/com.termux/files/home/eidosian_forge")).resolve()
for extra in (
    FORGE_ROOT / "lib",
    FORGE_ROOT / "doc_forge" / "src",
):
    text = str(extra)
    if extra.exists() and text not in sys.path:
        sys.path.insert(0, text)
DOC_RUNTIME = FORGE_ROOT / "doc_forge" / "runtime"
DOC_FINAL = DOC_RUNTIME / "final_docs"
DOC_INDEX = DOC_RUNTIME / "doc_index.json"
DOC_STATUS = DOC_RUNTIME / "processor_status.json"
RUNTIME_DIR = FORGE_ROOT / "data" / "runtime"
HOME_ROOT = Path(os.environ.get("HOME", "/data/data/com.termux/files/home")).resolve()
LOCAL_AGENT_STATUS = RUNTIME_DIR / "local_mcp_agent" / "status.json"
LOCAL_AGENT_HISTORY = RUNTIME_DIR / "local_mcp_agent" / "history.jsonl"
SCHEDULER_STATUS = RUNTIME_DIR / "eidos_scheduler_status.json"
COORDINATOR_STATUS = RUNTIME_DIR / "forge_coordinator_status.json"
COORDINATOR_HISTORY = RUNTIME_DIR / "forge_runtime_trends.json"
BOOT_STATUS = RUNTIME_DIR / "termux_boot_status.json"
CAPABILITIES_STATUS = RUNTIME_DIR / "platform_capabilities.json"
DIRECTORY_DOCS_STATUS = RUNTIME_DIR / "directory_docs_status.json"
DIRECTORY_DOCS_HISTORY = RUNTIME_DIR / "directory_docs_history.json"
DIRECTORY_DOCS_TREE = RUNTIME_DIR / "directory_docs_tree.json"
PROOF_REPORT_DIR = FORGE_ROOT / "reports" / "proof"
SERVICES_SCRIPT = FORGE_ROOT / "scripts" / "eidos_termux_services.sh"
SCHEDULER_CONTROL_SCRIPT = FORGE_ROOT / "scripts" / "eidos_scheduler_control.py"
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


def _docs_inventory(
    path_prefix: str = "",
    refresh: bool = False,
    missing_only: bool = False,
    suppressed_only: bool = False,
    review_only: bool = False,
    limit: int = 50,
) -> Dict[str, Any]:
    from doc_forge.scribe.directory_docs import inventory_summary, write_inventory_status  # type: ignore

    selected = {path_prefix.strip("/")} if path_prefix else None
    if not refresh and not path_prefix and not missing_only and not suppressed_only and not review_only and DIRECTORY_DOCS_STATUS.exists():
        status_payload = _read_json(DIRECTORY_DOCS_STATUS, {})
        if status_payload:
            result = dict(status_payload)
            result.setdefault("records", [])
            result["returned_count"] = 0
            return result

    if not path_prefix and not missing_only and not suppressed_only and not review_only:
        status_payload = write_inventory_status(FORGE_ROOT, DIRECTORY_DOCS_STATUS, selected_paths=selected)
    else:
        status_payload = {}
    payload = inventory_summary(FORGE_ROOT, selected_paths=selected)
    records = [row for row in payload.get("records", []) if isinstance(row, dict)]
    if missing_only:
        records = [row for row in records if not row.get("has_readme")]
    if suppressed_only:
        records = [row for row in records if row.get("suppressed")]
    if review_only:
        records = [row for row in records if row.get("review_required")]
    payload["records"] = records[: max(1, int(limit))]
    payload["returned_count"] = len(payload["records"])
    if status_payload:
        payload["status"] = status_payload
    return payload


def _docs_tree(path_prefix: str = "", limit: int = 250, refresh: bool = False) -> Dict[str, Any]:
    from doc_forge.scribe.directory_docs import inventory_tree  # type: ignore

    if not refresh and not path_prefix:
        cached = _read_json(DIRECTORY_DOCS_TREE, {})
        if cached:
            return cached
        status = _read_json(DIRECTORY_DOCS_STATUS, {})
        missing_examples = [row for row in (status.get("missing_examples") or []) if isinstance(row, str)]
        if missing_examples:
            nodes = [
                {
                    "path": row,
                    "name": Path(row).name,
                    "parent_path": "" if Path(row).parent.as_posix() == "." else Path(row).parent.as_posix(),
                    "depth": len(Path(row).parts),
                    "has_readme": False,
                    "tracked_files": 0,
                    "tests_present": False,
                    "api_route_count": 0,
                    "child_directory_count": 0,
                    "summary": "Cached missing README example from scheduler status.",
                }
                for row in missing_examples[: max(1, int(limit))]
            ]
            return {
                "contract": "eidos.documentation_tree.v1",
                "generated_at": status.get("generated_at", ""),
                "repo_root": str(FORGE_ROOT),
                "selected_paths": [],
                "returned_count": len(nodes),
                "groups": [],
                "nodes": nodes,
            }
    selected = {path_prefix.strip("/")} if path_prefix else None
    payload = inventory_tree(FORGE_ROOT, selected_paths=selected, limit=limit)
    if not path_prefix:
        _write_json(DIRECTORY_DOCS_TREE, payload)
    return payload


def get_docs_history(limit: int = 60) -> List[Dict[str, Any]]:
    payload = _read_json(DIRECTORY_DOCS_HISTORY, {})
    rows = payload.get("entries", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)][-max(1, int(limit)) :]


def get_runtime_snapshot() -> Dict[str, Any]:
    coordinator = _read_json(COORDINATOR_STATUS, {})
    scheduler = _read_json(SCHEDULER_STATUS, {})
    local_agent = _read_json(LOCAL_AGENT_STATUS, {})
    boot_status = _read_json(BOOT_STATUS, {})
    capabilities = _read_json(CAPABILITIES_STATUS, {})
    directory_docs = _read_json(DIRECTORY_DOCS_STATUS, {})
    if not capabilities:
        capabilities = asdict(collect_runtime_capabilities())
    return {
        "coordinator": coordinator,
        "scheduler": scheduler,
        "local_agent": local_agent,
        "boot": boot_status,
        "capabilities": capabilities,
        "directory_docs": directory_docs,
        "directory_docs_history": get_docs_history(limit=12),
    }


def get_local_agent_history(limit: int = 12) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not LOCAL_AGENT_HISTORY.exists():
        return rows
    try:
        lines = LOCAL_AGENT_HISTORY.read_text(encoding="utf-8").splitlines()
    except Exception:
        return rows
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
        if len(rows) >= max(1, int(limit)):
            break
    return rows


def get_runtime_history(limit: int = 24) -> List[Dict[str, Any]]:
    payload = _read_json(COORDINATOR_HISTORY, {})
    rows = payload.get("entries", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)][-max(1, int(limit)) :]


def get_latest_proof_report() -> Dict[str, Any]:
    latest = PROOF_REPORT_DIR / "entity_proof_scorecard_latest.json"
    payload = _read_json(latest, {})
    if not isinstance(payload, dict):
        return {}
    return payload


def get_file_tree(path: Path, root: Path) -> List[Dict[str, Any]]:
    tree = []
    try:
        # Sort directories first, then files
        entries = sorted(list(path.iterdir()), key=lambda x: (not x.is_dir(), x.name.lower()))
        for entry in entries:
            if entry.name.startswith(".") or entry.name == "__pycache__":
                continue
            item = {
                "name": entry.name,
                "path": str(entry.relative_to(root)),
                "is_dir": entry.is_dir(),
                "size": entry.stat().st_size if entry.is_file() else 0,
                "mtime": datetime.fromtimestamp(entry.stat().st_mtime).isoformat(),
            }
            if entry.is_dir():
                item["children"] = []  # Lazy loading not implemented for simplicity, just show dir
            tree.append(item)
    except Exception as e:
        logger.error(f"Error listing {path}: {e}")
    return tree


def _resolve_browse_root(domain: str) -> Path:
    if domain == "forge":
        return FORGE_ROOT
    if domain == "home":
        return HOME_ROOT
    return DOC_FINAL


def _service_command(action: str) -> Dict[str, Any]:
    allowed = {"start", "stop", "restart", "status"}
    if action not in allowed:
        raise HTTPException(status_code=400, detail="Invalid service action")
    if not SERVICES_SCRIPT.exists():
        raise HTTPException(status_code=503, detail="Service controller unavailable")
    env = os.environ.copy()
    env["EIDOS_FORGE_ROOT"] = str(FORGE_ROOT)
    result = subprocess.run(
        [str(SERVICES_SCRIPT), action],
        cwd=str(FORGE_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    return {
        "action": action,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "ok": result.returncode == 0,
        "services": _parse_service_status_output(result.stdout) if action == "status" else [],
    }


def _scheduler_command(action: str) -> Dict[str, Any]:
    allowed = {"status", "pause", "resume", "stop"}
    if action not in allowed:
        raise HTTPException(status_code=400, detail="Invalid scheduler action")
    if not SCHEDULER_CONTROL_SCRIPT.exists():
        raise HTTPException(status_code=503, detail="Scheduler controller unavailable")
    env = os.environ.copy()
    env["EIDOS_FORGE_ROOT"] = str(FORGE_ROOT)
    result = subprocess.run(
        [
            str(FORGE_ROOT / "eidosian_venv" / "bin" / "python"),
            str(SCHEDULER_CONTROL_SCRIPT),
            action,
        ],
        cwd=str(FORGE_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    payload: dict[str, Any] = {}
    try:
        payload = json.loads(result.stdout or "{}")
    except Exception:
        payload = {}
    return {
        "action": action,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "ok": result.returncode == 0,
        "payload": payload if isinstance(payload, dict) else {},
    }


def _parse_service_status_output(raw: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line or ":" not in line or line.startswith("Interactive shell refcount"):
            continue
        name, state = line.split(":", 1)
        state_value = state.strip()
        rows.append(
            {
                "name": name.strip(),
                "state": state_value,
                "running": "run:" in state_value or "running(" in state_value,
            }
        )
    return rows


# --- Routes ---


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    sys_stats = get_system_stats()
    forge_status = get_forge_status()
    doc_snapshot = get_doc_snapshot()
    docs_inventory = _docs_inventory(limit=12)
    docs_reviews = _docs_inventory(limit=12, review_only=True)
    docs_suppressed = _docs_inventory(limit=12, suppressed_only=True)
    docs_tree = _docs_tree(limit=40, refresh=False)
    docs_history = get_docs_history(limit=24)
    runtime_snapshot = get_runtime_snapshot()
    local_agent_history = get_local_agent_history()
    proof_snapshot = get_latest_proof_report()

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "request": request,
            "sys_stats": sys_stats,
            "forge_status": forge_status,
            "recent_docs": doc_snapshot["recent_docs"],
            "doc_status": doc_snapshot["status"],
            "doc_index_count": doc_snapshot["index_count"],
            "docs_inventory": docs_inventory,
            "docs_reviews": docs_reviews,
            "docs_suppressed": docs_suppressed,
            "docs_tree": docs_tree,
            "docs_history": docs_history,
            "runtime_snapshot": runtime_snapshot,
            "local_agent_history": local_agent_history,
            "proof_snapshot": proof_snapshot,
            "service_snapshot": _service_command("status"),
        },
    )


@app.get("/browse/{domain}/", response_class=HTMLResponse)
async def browse_domain_root(request: Request, domain: str):
    return await browse_domain(request, domain, ".")


@app.get("/browse/{domain}/{path:path}", response_class=HTMLResponse)
async def browse_domain(request: Request, domain: str, path: str = "."):
    root = _resolve_browse_root(domain)
    target_path = (root / path).resolve()
    if not str(target_path).startswith(str(DOC_FINAL.resolve())):
        if not str(target_path).startswith(str(root.resolve())):
            raise HTTPException(status_code=403, detail="Access denied")

    if domain == "docs" and not str(target_path).startswith(str(DOC_FINAL.resolve())):
        raise HTTPException(status_code=403, detail="Access denied")

    if not target_path.exists():
        raise HTTPException(status_code=404, detail="Path not found")

    if target_path.is_dir():
        files = get_file_tree(target_path, root)
        parent = str(Path(path).parent) if path != "." else None
        return templates.TemplateResponse(
            request,
            "browser.html",
            {
                "request": request,
                "domain": domain,
                "path": path,
                "files": files,
                "parent": parent,
                "root_label": root.name,
            },
        )
    elif target_path.is_file():
        if target_path.suffix.lower() == ".md":
            content = target_path.read_text()
            html_content = markdown.markdown(content, extensions=["fenced_code", "tables", "toc"])
            return templates.TemplateResponse(
                request,
                "viewer.html",
                {"request": request, "path": path, "content": html_content, "filename": target_path.name},
            )
        else:
            # For non-markdown files, just serve raw or download
            return FileResponse(target_path)


@app.get("/browse/{path:path}", response_class=HTMLResponse)
async def browse(request: Request, path: str):
    return await browse_domain(request, "docs", path)


@app.get("/api/system")
async def api_system():
    return get_system_stats()


@app.get("/api/doc/status")
async def api_doc_status():
    return get_doc_snapshot()


@app.get("/api/docs/coverage")
async def api_docs_coverage(
    limit: int = 50,
    missing_only: bool = False,
    suppressed_only: bool = False,
    review_only: bool = False,
    path_prefix: str = "",
    refresh: bool = False,
):
    return _docs_inventory(
        path_prefix=path_prefix,
        refresh=refresh,
        missing_only=missing_only,
        suppressed_only=suppressed_only,
        review_only=review_only,
        limit=limit,
    )


@app.get("/api/docs/tree")
async def api_docs_tree(limit: int = 250, path_prefix: str = "", refresh: bool = False):
    return _docs_tree(path_prefix=path_prefix, limit=limit, refresh=refresh)


@app.get("/api/docs/render")
async def api_docs_render(path: str):
    from doc_forge.scribe.directory_docs import render_directory_readme  # type: ignore

    return {"path": path, "content": render_directory_readme(FORGE_ROOT, path)}


@app.get("/api/docs/readme")
async def api_docs_readme(path: str):
    target = (FORGE_ROOT / path / "README.md").resolve()
    if not str(target).startswith(str(FORGE_ROOT.resolve())) or not target.exists():
        raise HTTPException(status_code=404, detail="README not found")
    return {"path": path, "content": target.read_text(encoding="utf-8")}


@app.get("/api/docs/diff")
async def api_docs_diff(path: str):
    from doc_forge.scribe.directory_docs import readme_diff  # type: ignore

    return {"path": path, "diff": readme_diff(FORGE_ROOT, path)}


@app.post("/api/docs/upsert")
async def api_docs_upsert(path: str):
    from doc_forge.scribe.directory_docs import upsert_directory_readme  # type: ignore

    return upsert_directory_readme(FORGE_ROOT, path)


@app.post("/api/docs/upsert-batch")
async def api_docs_upsert_batch(
    limit: int = 50,
    missing_only: bool = True,
    path_prefix: str = "",
    dry_run: bool = False,
):
    from doc_forge.scribe.directory_docs import upsert_directory_batch  # type: ignore

    return upsert_directory_batch(
        FORGE_ROOT,
        path_prefix=path_prefix,
        missing_only=missing_only,
        limit=limit,
        dry_run=dry_run,
    )


@app.post("/api/docs/refresh")
async def api_docs_refresh():
    return _docs_inventory(refresh=True, limit=20)


@app.get("/api/docs/history")
async def api_docs_history(limit: int = 60):
    return {"contract": "eidos.documentation_history.v1", "entries": get_docs_history(limit=limit)}


@app.get("/api/runtime")
async def api_runtime():
    snapshot = get_runtime_snapshot()
    snapshot["history"] = get_runtime_history()
    snapshot["proof"] = get_latest_proof_report()
    return snapshot


@app.get("/api/proof/latest")
async def api_proof_latest():
    payload = get_latest_proof_report()
    if payload:
        return payload
    raise HTTPException(status_code=404, detail="No proof report found")


@app.get("/api/services")
async def api_services():
    return _service_command("status")


@app.post("/api/services/{action}")
async def api_services_action(action: str):
    return _service_command(action)


@app.get("/api/capabilities")
async def api_capabilities():
    return get_runtime_snapshot().get("capabilities", {})


@app.get("/api/scheduler")
async def api_scheduler():
    return _scheduler_command("status")


@app.post("/api/scheduler/{action}")
async def api_scheduler_action(action: str):
    return _scheduler_command(action)


@app.get("/api/runtime/local-agent")
async def api_local_agent_status():
    return {
        "status": _read_json(LOCAL_AGENT_STATUS, {}),
        "history": get_local_agent_history(),
    }


@app.get("/health")
async def health():
    return {"status": "ok", "service": "eidos_atlas"}


if __name__ == "__main__":
    import uvicorn

    # Default port 8936 for dashboard (next to doc_forge 8930)
    port = int(os.environ.get("EIDOS_DASHBOARD_PORT", 8936))
    uvicorn.run(app, host="0.0.0.0", port=port)
