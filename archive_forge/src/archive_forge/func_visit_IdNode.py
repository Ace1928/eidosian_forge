from __future__ import annotations
import typing as T
def visit_IdNode(self, node: mparser.IdNode) -> None:
    self.visit_default_func(node)