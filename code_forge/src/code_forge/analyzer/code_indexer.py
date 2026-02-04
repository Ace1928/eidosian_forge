"""
Code Indexer for EIDOS.

Indexes Python codebases and stores structured code information
in the knowledge forge for semantic code search.
"""
from __future__ import annotations
from eidosian_core import eidosian

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .python_analyzer import CodeAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class CodeElement:
    """A code element (function, class, module)."""
    element_type: str  # "function", "class", "module", "method"
    name: str
    qualified_name: str  # Full path like module.class.method
    file_path: str
    line_start: int
    line_end: int
    docstring: Optional[str] = None
    source: Optional[str] = None
    args: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)  # For classes
    imports: List[str] = field(default_factory=list)  # For modules
    hash: str = field(default="")
    
    def __post_init__(self):
        if not self.hash and self.source:
            self.hash = hashlib.md5(self.source.encode()).hexdigest()[:12]
    
    @eidosian()
    def to_dict(self) -> Dict[str, Any]:
        return {
            "element_type": self.element_type,
            "name": self.name,
            "qualified_name": self.qualified_name,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "docstring": self.docstring,
            "args": self.args,
            "methods": self.methods,
            "imports": self.imports,
            "hash": self.hash,
        }


class CodeIndexer:
    """
    Indexes Python codebases for knowledge integration.
    
    Features:
    - AST-based code parsing
    - Element extraction (functions, classes, methods)
    - Change detection via content hashing
    - Knowledge forge integration
    """
    
    def __init__(self, index_path: Optional[Path] = None):
        self.analyzer = CodeAnalyzer()
        self.index_path = index_path or Path("/home/lloyd/eidosian_forge/data/code_index.json")
        self.elements: Dict[str, CodeElement] = {}
        self._knowledge_forge = None
        
        self._load_index()
    
    @property
    def knowledge(self):
        """Lazy-load knowledge forge."""
        if self._knowledge_forge is None:
            try:
                from knowledge_forge import KnowledgeForge
                kb_path = Path("/home/lloyd/eidosian_forge/data/kb.json")
                self._knowledge_forge = KnowledgeForge(persistence_path=kb_path)
            except ImportError as e:
                logger.warning(f"Could not import KnowledgeForge: {e}")
        return self._knowledge_forge
    
    @eidosian()
    def index_file(self, file_path: Path) -> List[CodeElement]:
        """Index a single Python file."""
        if not file_path.suffix == ".py":
            return []
        
        try:
            analysis = self.analyzer.analyze_file(file_path)
            if "error" in analysis:
                logger.warning(f"Analysis error for {file_path}: {analysis['error']}")
                return []
            
            elements: List[CodeElement] = []
            module_name = file_path.stem
            
            # Create module element
            source_text = file_path.read_text(encoding="utf-8")
            lines = source_text.splitlines()
            
            module_meta = analysis.get("module", {})
            module_elem = CodeElement(
                element_type="module",
                name=module_name,
                qualified_name=module_name,
                file_path=str(file_path),
                line_start=module_meta.get("line_start", 1),
                line_end=module_meta.get("line_end", len(lines)),
                docstring=module_meta.get("docstring"),
                source=module_meta.get("source"),
                imports=analysis.get("imports", []),
            )
            elements.append(module_elem)
            
            # Extract functions
            for func in analysis.get("functions", []):
                func_elem = CodeElement(
                    element_type="function",
                    name=func["name"],
                    qualified_name=f"{module_name}.{func['name']}",
                    file_path=str(file_path),
                    line_start=func.get("line_start"),
                    line_end=func.get("line_end"),
                    docstring=func.get("docstring"),
                    source=func.get("source"),
                    args=func.get("args", []),
                )
                elements.append(func_elem)
            
            # Extract classes and their methods
            for cls in analysis.get("classes", []):
                cls_elem = CodeElement(
                    element_type="class",
                    name=cls["name"],
                    qualified_name=f"{module_name}.{cls['name']}",
                    file_path=str(file_path),
                    line_start=cls.get("line_start"),
                    line_end=cls.get("line_end"),
                    docstring=cls.get("docstring"),
                    source=cls.get("source"),
                    methods=cls.get("methods", []),
                )
                elements.append(cls_elem)
                
                # Add individual methods
                for method_name in cls.get("methods", []):
                    method_elem = CodeElement(
                        element_type="method",
                        name=method_name,
                        qualified_name=f"{module_name}.{cls['name']}.{method_name}",
                        file_path=str(file_path),
                        line_start=cls.get("line_start"),
                        line_end=cls.get("line_end"),
                    )
                    elements.append(method_elem)
            
            # Store in index
            for elem in elements:
                self.elements[elem.qualified_name] = elem
            
            return elements
            
        except Exception as e:
            logger.error(f"Failed to index {file_path}: {e}")
            return []
    
    @eidosian()
    def index_directory(
        self,
        directory: Path,
        recursive: bool = True,
        exclude_patterns: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """Index all Python files in a directory."""
        exclude_patterns = exclude_patterns or ["__pycache__", ".git", "venv", ".venv", "node_modules"]
        
        stats = {
            "files_indexed": 0,
            "modules": 0,
            "classes": 0,
            "functions": 0,
            "methods": 0,
            "errors": 0,
        }
        
        pattern = "**/*.py" if recursive else "*.py"
        
        for py_file in directory.glob(pattern):
            # Skip excluded paths
            if any(exc in str(py_file) for exc in exclude_patterns):
                continue
            
            try:
                elements = self.index_file(py_file)
                if elements:
                    stats["files_indexed"] += 1
                    for elem in elements:
                        stats[elem.element_type + "s"] = stats.get(elem.element_type + "s", 0) + 1
            except Exception as e:
                logger.warning(f"Error indexing {py_file}: {e}")
                stats["errors"] += 1
        
        self._save_index()
        return stats
    
    @eidosian()
    def search(self, query: str, element_types: Optional[List[str]] = None) -> List[CodeElement]:
        """Search the code index."""
        query_lower = query.lower()
        results: List[CodeElement] = []
        
        for elem in self.elements.values():
            if element_types and elem.element_type not in element_types:
                continue
            
            # Match on name, qualified name, or docstring
            if (query_lower in elem.name.lower() or 
                query_lower in elem.qualified_name.lower() or
                (elem.docstring and query_lower in elem.docstring.lower())):
                results.append(elem)
        
        return results
    
    @eidosian()
    def sync_to_knowledge(self) -> int:
        """Sync indexed code elements to knowledge forge."""
        if not self.knowledge:
            return 0
        
        synced = 0
        for elem in self.elements.values():
            # Create knowledge content
            content = f"[CODE: {elem.element_type.upper()}] {elem.qualified_name}\n"
            if elem.docstring:
                content += f"Documentation: {elem.docstring}\n"
            if elem.args:
                content += f"Arguments: {', '.join(elem.args)}\n"
            if elem.methods:
                content += f"Methods: {', '.join(elem.methods)}\n"
            content += f"File: {elem.file_path}"
            
            # Add to knowledge forge
            self.knowledge.add_knowledge(
                content=content,
                concepts=[elem.element_type, "code", elem.name],
                tags=["code", elem.element_type, elem.name],
                metadata={
                    "source": "code_indexer",
                    "qualified_name": elem.qualified_name,
                    "file_path": elem.file_path,
                    "hash": elem.hash,
                },
            )
            synced += 1
        
        return synced
    
    @eidosian()
    def stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        type_counts: Dict[str, int] = {}
        for elem in self.elements.values():
            type_counts[elem.element_type] = type_counts.get(elem.element_type, 0) + 1
        
        return {
            "total_elements": len(self.elements),
            "by_type": type_counts,
            "index_path": str(self.index_path),
        }
    
    def _load_index(self) -> None:
        """Load index from disk."""
        if self.index_path.exists():
            try:
                with open(self.index_path) as f:
                    data = json.load(f)
                for qname, elem_data in data.items():
                    self.elements[qname] = CodeElement(**elem_data)
            except Exception as e:
                logger.warning(f"Failed to load index: {e}")
    
    def _save_index(self) -> None:
        """Save index to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        data = {qname: elem.to_dict() for qname, elem in self.elements.items()}
        with open(self.index_path, "w") as f:
            json.dump(data, f, indent=2)


# Convenience function
@eidosian()
def index_forge_codebase() -> Dict[str, int]:
    """Index the entire eidosian_forge codebase."""
    indexer = CodeIndexer()
    forge_root = Path("/home/lloyd/eidosian_forge")
    
    stats = indexer.index_directory(
        forge_root,
        recursive=True,
        exclude_patterns=[
            "__pycache__", ".git", "venv", ".venv", "node_modules",
            "google-cloud-sdk", "archive_forge", ".mypy_cache", ".pytest_cache"
        ],
    )
    
    return stats
