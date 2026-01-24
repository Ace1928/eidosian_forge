"""
Tika Content Extraction Router for EIDOS MCP Server.

Provides tools for extracting text and metadata from documents
using Apache Tika, with caching and knowledge forge integration.
"""

from __future__ import annotations
from eidosian_core import eidosian

import json
import os
from pathlib import Path
from typing import Optional, List

from ..core import tool
from ..forge_loader import ensure_forge_import

ensure_forge_import("crawl_forge")
ensure_forge_import("knowledge_forge")

try:
    from crawl_forge import TikaExtractor, TikaKnowledgeIngester
    TIKA_AVAILABLE = True
except ImportError:
    TIKA_AVAILABLE = False
    TikaExtractor = None
    TikaKnowledgeIngester = None

try:
    from knowledge_forge.core.graph import KnowledgeForge
except ImportError:
    KnowledgeForge = None

FORGE_DIR = Path(os.environ.get("EIDOS_FORGE_DIR", "/home/lloyd/eidosian_forge"))
TIKA_CACHE_DIR = Path(
    os.environ.get("EIDOS_TIKA_CACHE_DIR", Path.home() / ".eidosian" / "tika_cache")
)
KB_PATH = FORGE_DIR / "data" / "kb.json"

# Initialize components
_tika: Optional[TikaExtractor] = None
_knowledge_forge: Optional[KnowledgeForge] = None
_ingester: Optional[TikaKnowledgeIngester] = None


def _get_tika() -> Optional[TikaExtractor]:
    """Lazy-load the Tika extractor."""
    global _tika
    if _tika is None and TIKA_AVAILABLE:
        _tika = TikaExtractor(cache_dir=TIKA_CACHE_DIR, enable_cache=True)
    return _tika


def _get_ingester() -> Optional[TikaKnowledgeIngester]:
    """Lazy-load the knowledge ingester."""
    global _ingester, _knowledge_forge
    if _ingester is None and TIKA_AVAILABLE:
        tika = _get_tika()
        if tika:
            if _knowledge_forge is None and KnowledgeForge:
                _knowledge_forge = KnowledgeForge(persistence_path=KB_PATH)
            _ingester = TikaKnowledgeIngester(tika=tika, knowledge_forge=_knowledge_forge)
    return _ingester


# =============================================================================
# EXTRACTION TOOLS
# =============================================================================


@tool(
    name="tika_extract_file",
    description="Extract text and metadata from a local file using Apache Tika.",
    parameters={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to extract"
            },
            "use_cache": {
                "type": "boolean",
                "description": "Whether to use cached results (default: true)"
            },
        },
        "required": ["file_path"],
    },
)
@eidosian()
def tika_extract_file(file_path: str, use_cache: bool = True) -> str:
    """Extract text and metadata from a local file."""
    tika = _get_tika()
    if not tika:
        return "Error: Tika not available. Install with: pip install tika"
    
    result = tika.extract_from_file(Path(file_path), use_cache=use_cache)
    
    # Truncate content for display if very long
    if result.get("content") and len(result["content"]) > 5000:
        result["content"] = result["content"][:5000] + "\n... [truncated, use tika_ingest_file for full processing]"
    
    return json.dumps(result, indent=2, default=str)


@tool(
    name="tika_extract_url",
    description="Extract text and metadata from a URL using Apache Tika.",
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to extract content from"
            },
            "use_cache": {
                "type": "boolean",
                "description": "Whether to use cached results (default: true)"
            },
        },
        "required": ["url"],
    },
)
@eidosian()
def tika_extract_url(url: str, use_cache: bool = True) -> str:
    """Extract text and metadata from a URL."""
    tika = _get_tika()
    if not tika:
        return "Error: Tika not available. Install with: pip install tika"
    
    result = tika.extract_from_url(url, use_cache=use_cache)
    
    # Truncate content for display if very long
    if result.get("content") and len(result["content"]) > 5000:
        result["content"] = result["content"][:5000] + "\n... [truncated, use tika_ingest_url for full processing]"
    
    return json.dumps(result, indent=2, default=str)


@tool(
    name="tika_get_metadata",
    description="Get only metadata from a file (faster than full extraction).",
    parameters={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file"
            },
        },
        "required": ["file_path"],
    },
)
@eidosian()
def tika_get_metadata(file_path: str) -> str:
    """Get metadata from a file without full content extraction."""
    tika = _get_tika()
    if not tika:
        return "Error: Tika not available"
    
    metadata = tika.get_metadata_only(Path(file_path))
    return json.dumps(metadata, indent=2, default=str)


# =============================================================================
# KNOWLEDGE INGESTION TOOLS
# =============================================================================


@tool(
    name="tika_ingest_file",
    description="Extract a file's content and ingest it into the Knowledge Forge.",
    parameters={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to ingest"
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Tags to apply to the knowledge nodes"
            },
            "chunk_size": {
                "type": "integer",
                "description": "Maximum characters per chunk (default: 2000)"
            },
        },
        "required": ["file_path"],
    },
)
@eidosian()
def tika_ingest_file(
    file_path: str,
    tags: Optional[List[str]] = None,
    chunk_size: int = 2000,
) -> str:
    """Extract and ingest a file into Knowledge Forge."""
    ingester = _get_ingester()
    if not ingester:
        return "Error: Knowledge ingester not available"
    
    result = ingester.ingest_file(
        Path(file_path),
        tags=tags or [],
        chunk_size=chunk_size,
    )
    return json.dumps(result, indent=2, default=str)


@tool(
    name="tika_ingest_url",
    description="Extract a URL's content and ingest it into the Knowledge Forge.",
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to ingest"
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Tags to apply to the knowledge nodes"
            },
            "chunk_size": {
                "type": "integer",
                "description": "Maximum characters per chunk (default: 2000)"
            },
        },
        "required": ["url"],
    },
)
@eidosian()
def tika_ingest_url(
    url: str,
    tags: Optional[List[str]] = None,
    chunk_size: int = 2000,
) -> str:
    """Extract and ingest a URL into Knowledge Forge."""
    ingester = _get_ingester()
    if not ingester:
        return "Error: Knowledge ingester not available"
    
    result = ingester.ingest_url(
        url,
        tags=tags or [],
        chunk_size=chunk_size,
    )
    return json.dumps(result, indent=2, default=str)


@tool(
    name="tika_ingest_directory",
    description="Recursively ingest all documents from a directory.",
    parameters={
        "type": "object",
        "properties": {
            "directory": {
                "type": "string",
                "description": "Directory path to scan"
            },
            "extensions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "File extensions to include (e.g., ['pdf', 'docx'])"
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Tags to apply to all ingested nodes"
            },
            "recursive": {
                "type": "boolean",
                "description": "Whether to scan subdirectories (default: true)"
            },
        },
        "required": ["directory"],
    },
)
@eidosian()
def tika_ingest_directory(
    directory: str,
    extensions: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    recursive: bool = True,
) -> str:
    """Ingest all documents from a directory."""
    ingester = _get_ingester()
    if not ingester:
        return "Error: Knowledge ingester not available"
    
    dir_path = Path(directory)
    if not dir_path.exists():
        return f"Error: Directory not found: {directory}"
    
    # Default to common document types
    if not extensions:
        extensions = ["pdf", "doc", "docx", "txt", "md", "html", "htm", "rtf", "odt"]
    
    # Normalize extensions
    extensions = [ext.lower().lstrip('.') for ext in extensions]
    
    # Find files
    results = {
        "directory": str(dir_path),
        "files_found": 0,
        "files_processed": 0,
        "nodes_created": 0,
        "errors": [],
    }
    
    pattern = "**/*" if recursive else "*"
    for file_path in dir_path.glob(pattern):
        if file_path.is_file() and file_path.suffix.lstrip('.').lower() in extensions:
            results["files_found"] += 1
            
            file_result = ingester.ingest_file(
                file_path,
                tags=tags or [],
            )
            
            if file_result.get("status") == "success":
                results["files_processed"] += 1
                results["nodes_created"] += file_result.get("nodes_created", 0)
            elif file_result.get("status") == "error":
                results["errors"].append({
                    "file": str(file_path),
                    "error": file_result.get("error"),
                })
    
    return json.dumps(results, indent=2, default=str)


# =============================================================================
# CACHE MANAGEMENT
# =============================================================================


@tool(
    name="tika_cache_stats",
    description="Get statistics about the Tika extraction cache.",
    parameters={"type": "object", "properties": {}},
)
@eidosian()
def tika_cache_stats() -> str:
    """Get cache statistics."""
    tika = _get_tika()
    if not tika:
        return "Error: Tika not available"
    
    stats = tika.cache_stats()
    return json.dumps(stats, indent=2)


@tool(
    name="tika_cache_clear",
    description="Clear the Tika extraction cache.",
    parameters={
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "Specific source URL/path to clear, or omit for all"
            },
        },
    },
)
@eidosian()
def tika_cache_clear(source: Optional[str] = None) -> str:
    """Clear the extraction cache."""
    tika = _get_tika()
    if not tika:
        return "Error: Tika not available"
    
    count = tika.clear_cache(source)
    if source:
        return f"Cleared cache for: {source}" if count else f"No cache entry for: {source}"
    return f"Cleared {count} cache entries"
