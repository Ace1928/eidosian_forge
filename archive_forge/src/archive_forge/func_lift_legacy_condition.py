from __future__ import annotations
import typing
from .expr import Expr, Var, Value, Unary, Binary, Cast
from ..types import CastKind, cast_kind
from .. import types
def lift_legacy_condition(condition: tuple[qiskit.circuit.Clbit | qiskit.circuit.ClassicalRegister, int], /) -> Expr:
    """Lift a legacy two-tuple equality condition into a new-style :class:`Expr`.

    Examples:
        Taking an old-style conditional instruction and getting an :class:`Expr` from its
        condition::

            from qiskit.circuit import ClassicalRegister
            from qiskit.circuit.library import HGate
            from qiskit.circuit.classical import expr

            cr = ClassicalRegister(2)
            instr = HGate().c_if(cr, 3)

            lifted = expr.lift_legacy_condition(instr.condition)
    """
    from qiskit.circuit import Clbit
    target, value = condition
    if isinstance(target, Clbit):
        bool_ = types.Bool()
        return Var(target, bool_) if value else Unary(Unary.Op.LOGIC_NOT, Var(target, bool_), bool_)
    left = Var(target, types.Uint(width=target.size))
    if value.bit_length() > target.size:
        left = Cast(left, types.Uint(width=value.bit_length()), implicit=True)
    right = Value(value, left.type)
    return Binary(Binary.Op.EQUAL, left, right, types.Bool())