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
def is_degenerate(identity_set, gate_identity):
    """Checks if a gate identity is a permutation of another identity.

    Parameters
    ==========

    identity_set : set
        A Python set with GateIdentity objects.
    gate_identity : GateIdentity
        The GateIdentity to check for existence in the set.

    Examples
    ========

    Check if the identity is a permutation of another identity:

    >>> from sympy.physics.quantum.identitysearch import (
    ...     GateIdentity, is_degenerate)
    >>> from sympy.physics.quantum.gate import X, Y, Z
    >>> x = X(0); y = Y(0); z = Z(0)
    >>> an_identity = GateIdentity(x, y, z)
    >>> id_set = {an_identity}
    >>> another_id = (y, z, x)
    >>> is_degenerate(id_set, another_id)
    True

    >>> another_id = (x, x)
    >>> is_degenerate(id_set, another_id)
    False
    """
    for an_id in identity_set:
        if gate_identity in an_id.equivalent_ids:
            return True
    return False