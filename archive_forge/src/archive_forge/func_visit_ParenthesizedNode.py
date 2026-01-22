from __future__ import annotations
import typing as T
def visit_ParenthesizedNode(self, node: mparser.ParenthesizedNode) -> None:
    self.visit_default_func(node)
    node.inner.accept(self)