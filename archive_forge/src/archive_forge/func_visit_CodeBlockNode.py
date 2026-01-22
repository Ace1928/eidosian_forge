from __future__ import annotations
import typing as T
def visit_CodeBlockNode(self, node: mparser.CodeBlockNode) -> None:
    self.visit_default_func(node)
    for i in node.lines:
        i.accept(self)