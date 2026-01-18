"""
Refactor Forge - Pattern extraction and semantic transformations.
Ensures behavior guarantees during code restructuring.
"""
import ast
from typing import Dict, Any, List, Optional, Union

class RefactorTransformer(ast.NodeTransformer):
    """
    Standard AST transformer for refactoring operations.
    """
    def __init__(self, rename_map: Optional[Dict[str, str]] = None, remove_docs: bool = False):
        self.rename_map = rename_map or {}
        self.remove_docs = remove_docs

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        if node.name in self.rename_map:
            node.name = self.rename_map[node.name]
        
        if self.remove_docs and ast.get_docstring(node):
            node.body = node.body[1:] if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, (ast.Str, ast.Constant)) else node.body
            
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id in self.rename_map:
            node.id = self.rename_map[node.id]
        return self.generic_visit(node)

class RefactorForge:
    """
    Core implementation for code refactoring and pattern extraction.
    """
    def transform(self, source: str, rename_map: Optional[Dict[str, str]] = None, remove_docs: bool = False) -> str:
        """Apply multiple transformations to source code."""
        tree = ast.parse(source)
        transformer = RefactorTransformer(rename_map, remove_docs)
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)
        return ast.unparse(new_tree)

    def rename_functions(self, source: str, rename_map: Dict[str, str]) -> str:
        return self.transform(source, rename_map=rename_map)

    def remove_docstrings(self, source: str) -> str:
        return self.transform(source, remove_docs=True)

    def extract_pattern(self, source: str, pattern_type: type) -> List[str]:
        """Extract code snippets matching a specific AST pattern."""
        tree = ast.parse(source)
        snippets = []
        for node in ast.walk(tree):
            if isinstance(node, pattern_type):
                snippets.append(ast.unparse(node).strip())
        return snippets
