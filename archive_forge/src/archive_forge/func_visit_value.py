from __future__ import annotations
import typing
from . import expr
def visit_value(self, node, /):
    return node.__class__ is self.other.__class__ and node.type == self.other.type and (node.value == self.other.value)