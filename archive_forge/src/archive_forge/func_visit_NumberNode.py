from __future__ import annotations
import typing as T
def visit_NumberNode(self, node: mparser.NumberNode) -> None:
    self.visit_default_func(node)