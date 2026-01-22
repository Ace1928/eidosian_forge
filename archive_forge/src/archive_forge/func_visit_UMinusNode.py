from __future__ import annotations
import typing as T
def visit_UMinusNode(self, node: mparser.UMinusNode) -> None:
    self.visit_default_func(node)
    node.value.accept(self)