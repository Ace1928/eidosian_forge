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
def is_reducible(circuit, nqubits, begin, end):
    """Determines if a circuit is reducible by checking
    if its subcircuits are scalar values.

    Parameters
    ==========

    circuit : Gate tuple
        A tuple of Gates representing a circuit.  The circuit to check
        if a gate identity is contained in a subcircuit.
    nqubits : int
        The number of qubits the circuit operates on.
    begin : int
        The leftmost gate in the circuit to include in a subcircuit.
    end : int
        The rightmost gate in the circuit to include in a subcircuit.

    Examples
    ========

    Check if the circuit can be reduced:

    >>> from sympy.physics.quantum.identitysearch import is_reducible
    >>> from sympy.physics.quantum.gate import X, Y, Z
    >>> x = X(0); y = Y(0); z = Z(0)
    >>> is_reducible((x, y, z), 1, 0, 3)
    True

    Check if an interval in the circuit can be reduced:

    >>> is_reducible((x, y, z), 1, 1, 3)
    False

    >>> is_reducible((x, y, y), 1, 1, 3)
    True
    """
    current_circuit = ()
    for ndx in reversed(range(begin, end)):
        next_gate = circuit[ndx]
        current_circuit = (next_gate,) + current_circuit
        if is_scalar_matrix(current_circuit, nqubits, False):
            return True
    return False