"""
Code Analyzer - AST-based source analysis.
"""
import ast
from pathlib import Path
from typing import Dict, Any, List

class CodeAnalyzer:
    """Analyzes Python source code using the AST module."""
    
    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Parse a file and return its structure."""
        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            lines = source.splitlines()
            return self._visit_node(tree, lines)
        except Exception as e:
            return {"error": str(e), "file": str(file_path)}

    def _visit_node(self, node: ast.AST, lines: List[str]) -> Dict[str, Any]:
        """Recursively visit AST nodes."""
        summary = {
            "classes": [],
            "functions": [],
            "imports": [],
            "docstring": ast.get_docstring(node)
        }
        
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                # Extract source
                start = child.lineno - 1
                end = child.end_lineno if hasattr(child, "end_lineno") else child.lineno
                cls_source = "\n".join(lines[start:end])
                
                summary["classes"].append({
                    "name": child.name,
                    "docstring": ast.get_docstring(child),
                    "source": cls_source,
                    "methods": [n.name for n in child.body if isinstance(n, ast.FunctionDef)]
                })
            elif isinstance(child, ast.FunctionDef):
                # Extract source
                start = child.lineno - 1
                end = child.end_lineno if hasattr(child, "end_lineno") else child.lineno
                func_source = "\n".join(lines[start:end])
                
                summary["functions"].append({
                    "name": child.name,
                    "docstring": ast.get_docstring(child),
                    "source": func_source,
                    "args": [a.arg for a in child.args.args]
                })
            elif isinstance(child, (ast.Import, ast.ImportFrom)):
                # Simplified import handling
                if isinstance(child, ast.Import):
                    names = [n.name for n in child.names]
                else:
                    names = [f"{child.module}.{n.name}" for n in child.names]
                summary["imports"].extend(names)
                
        return summary
