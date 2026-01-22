from __future__ import annotations
import typing as T
def visit_MethodNode(self, node: mparser.MethodNode) -> None:
    self.visit_default_func(node)
    node.source_object.accept(self)
    node.name.accept(self)
    node.args.accept(self)