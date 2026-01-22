from __future__ import annotations
import typing
from .expr import Expr, Var, Value, Unary, Binary, Cast
from ..types import CastKind, cast_kind
from .. import types
def bit_not(operand: typing.Any, /) -> Expr:
    """Create a bitwise 'not' expression node from the given value, resolving any implicit casts and
    lifting the value into a :class:`Value` node if required.

    Examples:
        Bitwise negation of a :class:`.ClassicalRegister`::

            >>> from qiskit.circuit import ClassicalRegister
            >>> from qiskit.circuit.classical import expr
            >>> expr.bit_not(ClassicalRegister(3, "c"))
            Unary(Unary.Op.BIT_NOT, Var(ClassicalRegister(3, 'c'), Uint(3)), Uint(3))
    """
    operand = lift(operand)
    if operand.type.kind not in (types.Bool, types.Uint):
        raise TypeError(f"cannot apply '{Unary.Op.BIT_NOT}' to type '{operand.type}'")
    return Unary(Unary.Op.BIT_NOT, operand, operand.type)