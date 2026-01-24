from eidosian_core import eidosian
#!/usr/bin/env python3
"""
Validates the integrity of the context/index.json and context/catalog.json files.
Uses Pydantic for strict schema validation and performs logical consistency checks.
"""

import json
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, ValidationError

from tqdm import tqdm

# Setup Paths
ROOT_DIR = Path(__file__).resolve().parents[1]
CONTEXT_DIR = ROOT_DIR / "context"
INDEX_PATH = CONTEXT_DIR / "index.json"
CATALOG_PATH = CONTEXT_DIR / "catalog.json"
REPORT_DIR = ROOT_DIR / "home_audit_report"
REPORT_PATH = REPORT_DIR / "context_validation_report.md"

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("context_validator")

# --- Pydantic Models ---

class FileStats(BaseModel):
    size_bytes: Optional[int]
    last_modified: str
    mode: str
    is_symlink: bool
    owner: Optional[Any]
    group: Optional[Any]

class ChildSummary(BaseModel):
    count: Optional[int]
    preview_limit: int
    preview: List[Dict[str, Any]]
    preview_truncated: bool

class DirectoryEntry(BaseModel):
    name: str
    relative_path: str
    absolute_path: str
    type: str
    exists: bool
    section_names: List[str]
    manual: Dict[str, Any]
    statistics: FileStats
    scan_errors: List[str]
    child_summary: Optional[ChildSummary] = None
    git: Optional[Dict[str, str]] = None
    pyvenv_marker: Optional[str] = None

class IndexMetadata(BaseModel):
    root: str
    user: str
    hostname: str
    python_version: str
    config: Dict[str, Any]

class ContextIndex(BaseModel):
    generated_at: str
    generated_by: str
    metadata: IndexMetadata
    performance: Dict[str, Any]
    storage: Dict[str, Any]
    sections: List[Dict[str, Any]]
    directories: List[DirectoryEntry]

class SummaryMeta(BaseModel):
    model: str
    duration_sec: float
    path: str
    tokens: Optional[int]
    prompt_length: Optional[int] = None
    error: Optional[str] = None

class CatalogEntry(BaseModel):
    name: str
    relative_path: str
    absolute_path: str
    depth: int
    type: str
    statistics: Dict[str, Any]
    hash: Optional[str] = None
    summary: Optional[str] = None
    summary_meta: Optional[SummaryMeta] = None
    children: Optional[List['CatalogEntry']] = None # Recursive

class ContextCatalog(BaseModel):
    generated_at: str
    root: str
    catalog_config: Dict[str, Any]
    entries: List[CatalogEntry]
    summary_records: List[SummaryMeta]
    profiling: Dict[str, Any]

# --- Validation Logic ---

@eidosian()
def validate_paths_exist(entries: List[Any], base_path: Path) -> List[str]:
    errors = []
    for entry in tqdm(entries, desc="Verifying Paths"):
        # Handle both Index DirectoryEntry and CatalogEntry (recursive)
        path_str = getattr(entry, "absolute_path", None)
        if path_str:
            p = Path(path_str)
            if not p.exists():
                errors.append(f"Missing file/dir: {path_str}")
        
        # Recurse if children exist (CatalogEntry)
        if hasattr(entry, "children") and entry.children:
            errors.extend(validate_paths_exist(entry.children, base_path))
            
    return errors

@eidosian()
def flatten_catalog_entries(entries: List[CatalogEntry]) -> List[CatalogEntry]:
    flat = []
    for entry in entries:
        flat.append(entry)
        if entry.children:
            flat.extend(flatten_catalog_entries(entry.children))
    return flat

@eidosian()
def run_validation():
    logger.info("🔍 Starting Deep Context Validation...")
    report_lines = [f"# Context Validation Report - {datetime.now().isoformat()}", ""]
    
    # 1. Load Files
    index_raw = None
    catalog_raw = None
    
    try:
        index_raw = json.loads(INDEX_PATH.read_text())
        logger.info(f"✅ Loaded index.json ({len(index_raw.get('directories', []))}" + " roots)")
    except Exception as e:
        logger.error(f"❌ Failed to load index.json: {e}")
        report_lines.append(f"## ❌ Critical: index.json load failed\n{e}")
        return False

    try:
        catalog_raw = json.loads(CATALOG_PATH.read_text())
        logger.info(f"✅ Loaded catalog.json ({len(catalog_raw.get('entries', []))}" + " roots)")
    except Exception as e:
        logger.error(f"❌ Failed to load catalog.json: {e}")
        report_lines.append(f"## ❌ Critical: catalog.json load failed\n{e}")
        
    # 2. Schema Validation (Index)
    logger.info("Validating Index Schema...")
    try:
        index_model = ContextIndex(**index_raw)
        logger.info("✅ Index Schema Valid")
        report_lines.append("## ✅ Index Schema: Valid")
    except ValidationError as e:
        logger.error(f"❌ Index Schema Invalid: {e}")
        report_lines.append(f"## ❌ Index Schema: Invalid\n```\n{e}\n```")
        index_model = None

    # 3. Schema Validation (Catalog)
    logger.info("Validating Catalog Schema...")
    try:
        catalog_model = ContextCatalog(**catalog_raw)
        logger.info("✅ Catalog Schema Valid")
        report_lines.append("## ✅ Catalog Schema: Valid")
    except ValidationError as e:
        logger.error(f"❌ Catalog Schema Invalid: {e}")
        report_lines.append(f"## ❌ Catalog Schema: Invalid\n```\n{e}\n```")
        catalog_model = None

    # 4. Logical Checks
    if index_model:
        missing_paths = validate_paths_exist(index_model.directories, ROOT_DIR)
        if missing_paths:
            logger.warning(f"⚠️ {len(missing_paths)} paths in index do not exist on disk.")
            report_lines.append(f"## ⚠️ Index Path Integrity\n{len(missing_paths)} paths missing.")
        else:
            logger.info("✅ All index paths exist.")
            report_lines.append("## ✅ Index Path Integrity: Perfect")

    if catalog_model:
        all_catalog_items = flatten_catalog_entries(catalog_model.entries)
        missing_cat_paths = validate_paths_exist(all_catalog_items, ROOT_DIR)
        
        # Check summary coverage
        summarized = [e for e in all_catalog_items if e.summary]
        summary_rate = len(summarized) / len(all_catalog_items) if all_catalog_items else 0
        
        logger.info(f"Catalog Summary Coverage: {summary_rate:.1%} ({len(summarized)}/{len(all_catalog_items)})")
        report_lines.append(f"## ℹ️ Catalog Stats\n- Total Items: {len(all_catalog_items)}\n- Summarized: {len(summarized)}\n- Coverage: {summary_rate:.1%}")
        
        if missing_cat_paths:
             report_lines.append(f"## ⚠️ Catalog Path Integrity\n{len(missing_cat_paths)} paths missing.")
        else:
             report_lines.append("## ✅ Catalog Path Integrity: Perfect")

    # Save Report
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info(f"📝 Validation report saved to {REPORT_PATH}")
    
    return True

if __name__ == "__main__":
    try:
        run_validation()
    except KeyboardInterrupt:
        print("\nValidation interrupted.")
        sys.exit(130)
