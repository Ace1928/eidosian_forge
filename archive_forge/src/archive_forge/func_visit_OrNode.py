from __future__ import annotations
import typing as T
def visit_OrNode(self, node: mparser.OrNode) -> None:
    self.visit_default_func(node)
    node.left.accept(self)
    node.right.accept(self)