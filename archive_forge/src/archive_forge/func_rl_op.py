from collections import deque
from sympy.core.random import randint
from sympy.external import import_module
from sympy.core.basic import Basic
from sympy.core.mul import Mul
from sympy.core.numbers import Number, equal_valued
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.dagger import Dagger
def rl_op(left, right):
    """Perform a RL operation.

    A RL operation multiplies both left and right circuits
    with the dagger of the right circuit's leftmost gate, and
    the dagger is multiplied on the left side of both circuits.

    If a RL is possible, it returns the new gate rule as a
    2-tuple (LHS, RHS), where LHS is the left circuit and
    and RHS is the right circuit of the new rule.
    If a RL is not possible, None is returned.

    Parameters
    ==========

    left : Gate tuple
        The left circuit of a gate rule expression.
    right : Gate tuple
        The right circuit of a gate rule expression.

    Examples
    ========

    Generate a new gate rule using a RL operation:

    >>> from sympy.physics.quantum.identitysearch import rl_op
    >>> from sympy.physics.quantum.gate import X, Y, Z
    >>> x = X(0); y = Y(0); z = Z(0)
    >>> rl_op((x,), (y, z))
    ((Y(0), X(0)), (Z(0),))

    >>> rl_op((x, y), (z,))
    ((Z(0), X(0), Y(0)), ())
    """
    if len(right) > 0:
        rl_gate = right[0]
        rl_gate_is_unitary = is_scalar_matrix((Dagger(rl_gate), rl_gate), _get_min_qubits(rl_gate), True)
    if len(right) > 0 and rl_gate_is_unitary:
        new_right = right[1:len(right)]
        new_left = (Dagger(rl_gate),) + left
        return (new_left, new_right)
    return None