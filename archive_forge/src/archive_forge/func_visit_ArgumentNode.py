from __future__ import annotations
import typing as T
def visit_ArgumentNode(self, node: mparser.ArgumentNode) -> None:
    self.visit_default_func(node)
    for i in node.arguments:
        i.accept(self)
    for key, val in node.kwargs.items():
        key.accept(self)
        val.accept(self)