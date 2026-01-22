from __future__ import annotations
import typing as T
def visit_FormatStringNode(self, node: mparser.FormatStringNode) -> None:
    self.visit_default_func(node)