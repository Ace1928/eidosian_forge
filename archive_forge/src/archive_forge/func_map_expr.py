from __future__ import annotations
import typing
from .bit import Bit
from .classical import expr
from .classicalregister import ClassicalRegister, Clbit
def map_expr(self, node: expr.Expr, /) -> expr.Expr:
    """Map the variables in an :class:`~.expr.Expr` node to the new circuit."""
    return node.accept(self)