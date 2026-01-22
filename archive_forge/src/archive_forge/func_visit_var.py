from __future__ import annotations
import typing
from . import expr
def visit_var(self, node, /):
    if self.other.__class__ is not node.__class__ or self.other.type != node.type:
        return False
    if self.self_key is None or (self_var := self.self_key(node.var)) is None:
        self_var = node.var
    if self.other_key is None or (other_var := self.other_key(self.other.var)) is None:
        other_var = self.other.var
    return self_var == other_var