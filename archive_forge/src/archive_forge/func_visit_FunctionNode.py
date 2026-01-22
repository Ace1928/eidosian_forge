from __future__ import annotations
import typing as T
def visit_FunctionNode(self, node: mparser.FunctionNode) -> None:
    self.visit_default_func(node)
    node.func_name.accept(self)
    node.args.accept(self)