import markdown
import logging
from pathlib import Path
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from ..config import DOC_FINAL, FORGE_ROOT, HOME_ROOT
from ..templating import templates
from datetime import datetime

router = APIRouter()
logger = logging.getLogger("eidos_dashboard")

def get_file_tree(path: Path, root: Path) -> list:
    tree = []
    try:
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
            tree.append(item)
    except Exception as e:
        logger.error(f"Error listing {path}: {e}")
    return tree

def _resolve_browse_root(domain: str) -> Path:
    if domain == "forge":
        return FORGE_ROOT
    if domain == "home":
        return HOME_ROOT
    if domain == "knowledge":
        return FORGE_ROOT / "knowledge_forge"
    if domain == "word":
        return FORGE_ROOT / "word_forge"
    return DOC_FINAL

@router.get("/browse/{domain}", response_class=HTMLResponse)
async def browse_domain_no_slash(request: Request, domain: str):
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url=f"/browse/{domain}/")

@router.get("/browse/{domain}/", response_class=HTMLResponse)
async def browse_domain_root(request: Request, domain: str):
    return await browse_domain(request, domain, ".")

@router.get("/browse/{domain}/{path:path}", response_class=HTMLResponse)
async def browse_domain(request: Request, domain: str, path: str = "."):
    root = _resolve_browse_root(domain)
    target_path = (root / path).resolve()
    
    # Security: Ensure path is within the domain root
    if not str(target_path).startswith(str(root.resolve())):
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
                "domain": domain,
                "path": path,
                "files": files,
                "parent": parent,
                "root_label": root.name,
            },
        )
    elif target_path.is_file():
        if target_path.suffix.lower() == ".md":
            content = target_path.read_text(encoding="utf-8", errors="replace")
            html_content = markdown.markdown(content, extensions=["fenced_code", "tables", "toc"])
            return templates.TemplateResponse(
                request,
                "viewer.html",
                {"path": path, "content": html_content, "filename": target_path.name},
            )
        else:
            return FileResponse(target_path)

@router.get("/browse/{path:path}", response_class=HTMLResponse)
async def browse(request: Request, path: str):
    return await browse_domain(request, "docs", path)
