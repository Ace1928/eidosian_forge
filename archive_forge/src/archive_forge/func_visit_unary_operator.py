from __future__ import annotations
from .. import mparser
from .visitor import AstVisitor
from itertools import zip_longest
import re
import typing as T
def visit_unary_operator(self, node: mparser.UnaryOperatorNode) -> None:
    node.operator.accept(self)
    node.value.accept(self)
    if node.whitespaces:
        node.whitespaces.accept(self)