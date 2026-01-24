"""Refactor forge core surface area."""
from __future__ import annotations

import ast
from typing import Dict, Optional
from eidosian_core import eidosian

__version__ = "0.1.0"


class _RenameTransformer(ast.NodeTransformer):
    def __init__(self, rename_map: Dict[str, str]) -> None:
        self._rename_map = rename_map

    @eidosian()
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        if node.name in self._rename_map:
            node.name = self._rename_map[node.name]
        self.generic_visit(node)
        return node

    @eidosian()
    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self._rename_map:
            node.id = self._rename_map[node.id]
        return node


def _strip_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if isinstance(body[0].value.value, str):
            return body[1:]
    return body


class RefactorForge:
    """Minimal refactor utilities for tests and MCP use."""

    @eidosian()
    def transform(
        self,
        source: str,
        rename_map: Optional[Dict[str, str]] = None,
        remove_docs: bool = False,
    ) -> str:
        tree = ast.parse(source)
        if remove_docs:
            tree.body = _strip_docstring(tree.body)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    node.body = _strip_docstring(node.body)
        if rename_map:
            tree = _RenameTransformer(rename_map).visit(tree)
            ast.fix_missing_locations(tree)
        return ast.unparse(tree)


__all__ = ["RefactorForge", "__version__"]
