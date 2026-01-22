from __future__ import annotations
import typing
from . import expr
def iter_vars(node: expr.Expr) -> typing.Iterator[expr.Var]:
    """Get an iterator over the :class:`~.expr.Var` nodes referenced at any level in the given
    :class:`~.expr.Expr`.

    Examples:
        Print out the name of each :class:`.ClassicalRegister` encountered::

            from qiskit.circuit import ClassicalRegister
            from qiskit.circuit.classical import expr

            cr1 = ClassicalRegister(3, "a")
            cr2 = ClassicalRegister(3, "b")

            for node in expr.iter_vars(expr.bit_and(expr.bit_not(cr1), cr2)):
                if isinstance(node.var, ClassicalRegister):
                    print(node.var.name)
    """
    yield from node.accept(_VAR_WALKER)