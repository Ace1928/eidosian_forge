from __future__ import annotations
import typing as T
def visit_StringNode(self, node: mparser.StringNode) -> None:
    self.visit_default_func(node)