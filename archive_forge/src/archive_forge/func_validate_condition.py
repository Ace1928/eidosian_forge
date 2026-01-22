from __future__ import annotations
import dataclasses
from typing import Iterable, Tuple, Set, Union, TypeVar, TYPE_CHECKING
from qiskit.circuit.classical import expr, types
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.register import Register
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.circuit.quantumregister import QuantumRegister
def validate_condition(condition: _ConditionT) -> _ConditionT:
    """Validate that a condition is in a valid format and return it, but raise if it is invalid.

    Args:
        condition: the condition to be tested for validity.  Must be either the legacy 2-tuple
            format, or a :class:`~.expr.Expr` that has `Bool` type.

    Raises:
        CircuitError: if the condition is not in a valid format.

    Returns:
        The same condition as passed, if it was valid.
    """
    if isinstance(condition, expr.Expr):
        if condition.type.kind is not types.Bool:
            raise CircuitError(f"Classical conditions must be expressions with the type 'Bool()', not '{condition.type}'.")
        return condition
    try:
        bits, value = condition
        if isinstance(bits, (ClassicalRegister, Clbit)) and isinstance(value, int):
            return (bits, value)
    except (TypeError, ValueError):
        pass
    raise CircuitError(f"A classical condition should be a 2-tuple of `(ClassicalRegister | Clbit, int)`, but received '{condition!r}'.")