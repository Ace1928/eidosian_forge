from __future__ import annotations
import typing
from .expr import Expr, Var, Value, Unary, Binary, Cast
from ..types import CastKind, cast_kind
from .. import types
def bit_or(left: typing.Any, right: typing.Any, /) -> Expr:
    """Create a bitwise 'or' expression node from the given value, resolving any implicit casts and
    lifting the values into :class:`Value` nodes if required.

    Examples:
        Bitwise 'or' of a classical register and an integer literal::

            >>> from qiskit.circuit import ClassicalRegister
            >>> from qiskit.circuit.classical import expr
            >>> expr.bit_or(ClassicalRegister(3, "c"), 0b101)
            Binary(Binary.Op.BIT_OR, Var(ClassicalRegister(3, 'c'), Uint(3)), Value(5, Uint(3)), Uint(3))
    """
    return _binary_bitwise(Binary.Op.BIT_OR, left, right)