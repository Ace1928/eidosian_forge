from functools import reduce
from sympy.core.sorting import default_sort_key
from sympy.core.containers import Tuple
from sympy.core.mul import Mul
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.utilities import numbered_symbols
from sympy.physics.quantum.gate import Gate
def find_subcircuit(circuit, subcircuit, start=0, end=0):
    """Finds the subcircuit in circuit, if it exists.

    Explanation
    ===========

    If the subcircuit exists, the index of the start of
    the subcircuit in circuit is returned; otherwise,
    -1 is returned.  The algorithm that is implemented
    is the Knuth-Morris-Pratt algorithm.

    Parameters
    ==========

    circuit : tuple, Gate or Mul
        A tuple of Gates or Mul representing a quantum circuit
    subcircuit : tuple, Gate or Mul
        A tuple of Gates or Mul to find in circuit
    start : int
        The location to start looking for subcircuit.
        If start is the same or past end, -1 is returned.
    end : int
        The last place to look for a subcircuit.  If end
        is less than 1 (one), then the length of circuit
        is taken to be end.

    Examples
    ========

    Find the first instance of a subcircuit:

    >>> from sympy.physics.quantum.circuitutils import find_subcircuit
    >>> from sympy.physics.quantum.gate import X, Y, Z, H
    >>> circuit = X(0)*Z(0)*Y(0)*H(0)
    >>> subcircuit = Z(0)*Y(0)
    >>> find_subcircuit(circuit, subcircuit)
    1

    Find the first instance starting at a specific position:

    >>> find_subcircuit(circuit, subcircuit, start=1)
    1

    >>> find_subcircuit(circuit, subcircuit, start=2)
    -1

    >>> circuit = circuit*subcircuit
    >>> find_subcircuit(circuit, subcircuit, start=2)
    4

    Find the subcircuit within some interval:

    >>> find_subcircuit(circuit, subcircuit, start=2, end=2)
    -1
    """
    if isinstance(circuit, Mul):
        circuit = circuit.args
    if isinstance(subcircuit, Mul):
        subcircuit = subcircuit.args
    if len(subcircuit) == 0 or len(subcircuit) > len(circuit):
        return -1
    if end < 1:
        end = len(circuit)
    pos = start
    index = 0
    table = kmp_table(subcircuit)
    while pos + index < end:
        if subcircuit[index] == circuit[pos + index]:
            index = index + 1
        else:
            pos = pos + index - table[index]
            index = table[index] if table[index] > -1 else 0
        if index == len(subcircuit):
            return pos
    return -1