from __future__ import annotations
import typing
from .expr import Expr, Var, Value, Unary, Binary, Cast
from ..types import CastKind, cast_kind
from .. import types
def logic_or(left: typing.Any, right: typing.Any, /) -> Expr:
    """Create a logical 'or' expression node from the given value, resolving any implicit casts and
    lifting the values into :class:`Value` nodes if required.

    Examples:
        Logical 'or' of two classical bits

            >>> from qiskit.circuit import Clbit
            >>> from qiskit.circuit.classical import expr
            >>> expr.logical_and(Clbit(), Clbit())
            Binary(Binary.Op.LOGIC_OR, Var(<clbit 0>, Bool()), Var(<clbit 1>, Bool()), Bool())
    """
    return _binary_logical(Binary.Op.LOGIC_OR, left, right)