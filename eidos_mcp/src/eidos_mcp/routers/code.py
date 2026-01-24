"""
Code Tools Router for EIDOS MCP Server.

Provides tools for code analysis, indexing, and search.
"""

from __future__ import annotations
from eidosian_core import eidosian

import json
import os
from pathlib import Path
from typing import List, Optional

from ..core import tool
from ..forge_loader import ensure_forge_import

ensure_forge_import("code_forge")

try:
    from code_forge import CodeIndexer, CodeAnalyzer, index_forge_codebase
    CODE_FORGE_AVAILABLE = True
except ImportError:
    CODE_FORGE_AVAILABLE = False
    CodeIndexer = None
    CodeAnalyzer = None

FORGE_DIR = Path(os.environ.get("EIDOS_FORGE_DIR", "/home/lloyd/eidosian_forge"))

# Lazy-loaded instances
_indexer = None
_analyzer = None


def _get_indexer():
    """Lazy-load the code indexer."""
    global _indexer
    if _indexer is None and CODE_FORGE_AVAILABLE:
        _indexer = CodeIndexer()
    return _indexer


def _get_analyzer():
    """Lazy-load the code analyzer."""
    global _analyzer
    if _analyzer is None and CODE_FORGE_AVAILABLE:
        _analyzer = CodeAnalyzer()
    return _analyzer


@tool(
    name="code_search",
    description="Search indexed code elements (functions, classes, methods) by name or docstring.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            },
            "element_types": {
                "type": "array",
                "items": {"type": "string", "enum": ["function", "class", "method", "module"]},
                "description": "Filter by element types (default: all)"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results (default: 10)"
            },
        },
        "required": ["query"],
    },
)
@eidosian()
def code_search(
    query: str,
    element_types: Optional[List[str]] = None,
    limit: int = 10,
) -> str:
    """Search the code index."""
    indexer = _get_indexer()
    if not indexer:
        return "Error: Code indexer not available"
    
    results = indexer.search(query, element_types=element_types)
    
    if not results:
        return f"No code elements found matching '{query}'"
    
    output = []
    for elem in results[:limit]:
        item = {
            "type": elem.element_type,
            "name": elem.name,
            "qualified_name": elem.qualified_name,
            "file": elem.file_path,
        }
        if elem.docstring:
            item["docstring"] = elem.docstring[:100] + "..." if len(elem.docstring) > 100 else elem.docstring
        if elem.args:
            item["args"] = elem.args
        if elem.methods:
            item["methods"] = elem.methods[:5]  # First 5 methods
        output.append(item)
    
    return json.dumps(output, indent=2)


@tool(
    name="code_analyze_file",
    description="Analyze a Python file and extract its structure (classes, functions, imports).",
    parameters={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the Python file to analyze"
            },
        },
        "required": ["file_path"],
    },
)
@eidosian()
def code_analyze_file(file_path: str) -> str:
    """Analyze a Python file."""
    analyzer = _get_analyzer()
    if not analyzer:
        return "Error: Code analyzer not available"
    
    path = Path(file_path)
    if not path.exists():
        return f"Error: File not found: {file_path}"
    
    if not path.suffix == ".py":
        return "Error: Only Python files (.py) are supported"
    
    analysis = analyzer.analyze_file(path)
    
    # Format output
    output = {
        "file": str(path),
        "docstring": analysis.get("docstring"),
        "imports": analysis.get("imports", []),
        "classes": [
            {
                "name": c["name"],
                "docstring": c.get("docstring"),
                "methods": c.get("methods", []),
            }
            for c in analysis.get("classes", [])
        ],
        "functions": [
            {
                "name": f["name"],
                "docstring": f.get("docstring"),
                "args": f.get("args", []),
            }
            for f in analysis.get("functions", [])
        ],
    }
    
    return json.dumps(output, indent=2)


@tool(
    name="code_index_stats",
    description="Get statistics about the code index.",
    parameters={"type": "object", "properties": {}},
)
@eidosian()
def code_index_stats() -> str:
    """Get code index statistics."""
    indexer = _get_indexer()
    if not indexer:
        return "Error: Code indexer not available"
    
    stats = indexer.stats()
    return json.dumps(stats, indent=2)


@tool(
    name="code_index_directory",
    description="Index Python files in a directory.",
    parameters={
        "type": "object",
        "properties": {
            "directory": {
                "type": "string",
                "description": "Directory path to index"
            },
            "recursive": {
                "type": "boolean",
                "description": "Search recursively (default: true)"
            },
        },
        "required": ["directory"],
    },
)
@eidosian()
def code_index_directory(directory: str, recursive: bool = True) -> str:
    """Index Python files in a directory."""
    indexer = _get_indexer()
    if not indexer:
        return "Error: Code indexer not available"
    
    path = Path(directory)
    if not path.exists():
        return f"Error: Directory not found: {directory}"
    
    stats = indexer.index_directory(path, recursive=recursive)
    
    return json.dumps({
        "directory": str(path),
        "files_indexed": stats.get("files_indexed", 0),
        "modules": stats.get("modules", 0),
        "classes": stats.get("classs", 0),  # Note: typo in original
        "functions": stats.get("functions", 0),
        "methods": stats.get("methods", 0),
        "errors": stats.get("errors", 0),
    }, indent=2)


@tool(
    name="code_sync_to_knowledge",
    description="Sync indexed code elements to the knowledge forge for semantic search.",
    parameters={"type": "object", "properties": {}},
)
@eidosian()
def code_sync_to_knowledge() -> str:
    """Sync code index to knowledge forge."""
    indexer = _get_indexer()
    if not indexer:
        return "Error: Code indexer not available"
    
    synced = indexer.sync_to_knowledge()
    return f"Synced {synced} code elements to knowledge forge"
