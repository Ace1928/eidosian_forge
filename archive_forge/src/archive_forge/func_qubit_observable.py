import numpy as np
import pennylane as qml
from pennylane.fermi import FermiSentence, FermiWord
from pennylane.operation import active_new_opmath
from pennylane.pauli.utils import simplify
def qubit_observable(o_ferm, cutoff=1e-12):
    """Convert a fermionic observable to a PennyLane qubit observable.

    Args:
        o_ferm (Union[FermiWord, FermiSentence]): fermionic operator
        cutoff (float): cutoff value for discarding the negligible terms

    Returns:
        Operator: Simplified PennyLane Hamiltonian

    **Example**

    >>> w1 = qml.fermi.FermiWord({(0, 0) : '+', (1, 1) : '-'})
    >>> w2 = qml.fermi.FermiWord({(0, 1) : '+', (1, 2) : '-'})
    >>> s = qml.fermi.FermiSentence({w1 : 1.2, w2: 3.1})
    >>> print(qubit_observable(s))
      (-0.3j) [Y0 X1]
    + (0.3j) [X0 Y1]
    + (-0.775j) [Y1 X2]
    + (0.775j) [X1 Y2]
    + ((0.3+0j)) [Y0 Y1]
    + ((0.3+0j)) [X0 X1]
    + ((0.775+0j)) [Y1 Y2]
    + ((0.775+0j)) [X1 X2]

    If the new op-math is active, an arithmetic operator is returned.

    >>> qml.operation.enable_new_opmath()
    >>> w1 = qml.fermi.FermiWord({(0, 0) : '+', (1, 1) : '-'})
    >>> w2 = qml.fermi.FermiWord({(0, 0) : '+', (1, 1) : '-'})
    >>> s = qml.fermi.FermiSentence({w1 : 1.2, w2: 3.1})
    >>> print(qubit_observable(s))
    -0.775j * (Y(0) @ X(1)) + 0.775 * (Y(0) @ Y(1)) + 0.775 * (X(0) @ X(1)) + 0.775j * (X(0) @ Y(1))
    """
    h = qml.jordan_wigner(o_ferm, ps=True, tol=cutoff)
    h.simplify(tol=cutoff)
    if active_new_opmath():
        if not h.wires:
            return h.operation(wire_order=[0])
        return h.operation()
    if not h.wires:
        h = h.hamiltonian(wire_order=[0])
        return qml.Hamiltonian(h.coeffs, [qml.Identity(0) if o.name == 'Identity' else o for o in h.ops])
    h = h.hamiltonian()
    return simplify(qml.Hamiltonian(h.coeffs, [qml.Identity(0) if o.name == 'Identity' else o for o in h.ops]))