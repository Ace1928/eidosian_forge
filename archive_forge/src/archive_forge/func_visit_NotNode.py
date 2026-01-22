from __future__ import annotations
import typing as T
def visit_NotNode(self, node: mparser.NotNode) -> None:
    self.visit_default_func(node)
    node.value.accept(self)