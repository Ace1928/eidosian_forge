"""
Code Analyzer - Core Analysis Engine 🔍

Coordinates the complete code analysis process, integrating results from
specialized analyzers to build comprehensive structural understanding.
"""

from pathlib import Path
from typing import Dict, Any, Union

import networkx as nx

from ..core.types import AnalysisResult
from .module_detector import detect_module_boundaries
from .dependency_analyzer import build_dependency_graph
from .semantic_analyzer import extract_semantic_purpose
from .import_analyzer import analyze_imports


class CodeAnalyzer:
    """Advanced code analyzer that builds comprehensive structural understanding.
    
    Parses Python source into AST and extracts:
    - Structural patterns and natural boundaries
    - Dependency relationships and import requirements
    - Functional groups and their relationships
    - Semantic purpose through comment and docstring analysis
    """
    
    def __init__(self, source_path: Union[str, Path]):
        """Initialize analyzer with path to source file.
        
        Args:
            source_path: Path to Python file for analysis
        """
        self.source_path = Path(source_path)
        if not self.source_path.exists():
            raise FileNotFoundError(f"Source file not found: {self.source_path}")
            
        self.source_code = self.source_path.read_text(encoding='utf-8')
        self.symbol_table = {}
    
    def analyze(self) -> Dict[str, Any]:
        """Perform full structural analysis of the source code.
        
        Returns:
            Complete analysis results with structural insights
        """
        # Parse imports and build initial symbol table
        self.symbol_table = analyze_imports(self.source_code)
        
        # Detect logical boundaries in the code
        modules = detect_module_boundaries(self.source_code)
        
        # Build dependency graph between identified components
        dependencies = build_dependency_graph(modules)
        
        # Extract documentation and purpose for components
        modules_with_semantics = extract_semantic_purpose(modules)
        
        # Assemble the complete analysis result
        return {
            "modules": modules_with_semantics,
            "dependencies": dependencies,
            "symbols": self.symbol_table,
            "file_info": {
                "path": str(self.source_path),
                "size": self.source_path.stat().st_size,
                "name": self.source_path.name,
                "stem": self.source_path.stem,
            }
        }
