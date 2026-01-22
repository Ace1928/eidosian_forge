from __future__ import annotations
import typing
from . import expr
def structurally_equivalent(left: expr.Expr, right: expr.Expr, left_var_key: typing.Callable[[typing.Any], typing.Any] | None=None, right_var_key: typing.Callable[[typing.Any], typing.Any] | None=None) -> bool:
    """Do these two expressions have exactly the same tree structure, up to some key function for
    the :class:`~.expr.Var` objects?

    In other words, are these two expressions the exact same trees, except we compare the
    :attr:`.Var.var` fields by calling the appropriate ``*_var_key`` function on them, and comparing
    that output for equality.  This function does not allow any semantic "equivalences" such as
    asserting that ``a == b`` is equivalent to ``b == a``; the evaluation order of the operands
    could, in general, cause such a statement to be false (consider hypothetical ``extern``
    functions that access global state).

    There's no requirements on the key functions, except that their outputs should have general
    ``__eq__`` methods.  If a key function returns ``None``, the variable will be used verbatim
    instead.

    Args:
        left: one of the :class:`~.expr.Expr` nodes.
        right: the other :class:`~.expr.Expr` node.
        left_var_key: a callable whose output should be used when comparing :attr:`.Var.var`
            attributes.  If this argument is ``None`` or its output is ``None`` for a given
            variable in ``left``, the variable will be used verbatim.
        right_var_key: same as ``left_var_key``, but used on the variables in ``right`` instead.

    Examples:
        Comparing two expressions for structural equivalence, with no remapping of the variables.
        These are different because the different :class:`.Clbit` instances compare differently::

            >>> from qiskit.circuit import Clbit
            >>> from qiskit.circuit.classical import expr
            >>> left_bits = [Clbit(), Clbit()]
            >>> right_bits = [Clbit(), Clbit()]
            >>> left = expr.logic_and(expr.logic_not(left_bits[0]), left_bits[1])
            >>> right = expr.logic_and(expr.logic_not(right_bits[0]), right_bits[1])
            >>> expr.structurally_equivalent(left, right)
            False

        Comparing the same two expressions, but this time using mapping functions that associate
        the bits with simple indices::

            >>> left_key = {var: i for i, var in enumerate(left_bits)}.get
            >>> right_key = {var: i for i, var in enumerate(right_bits)}.get
            >>> expr.structurally_equivalent(left, right, left_key, right_key)
            True
    """
    return left.accept(_StructuralEquivalenceImpl(right, left_var_key, right_var_key))