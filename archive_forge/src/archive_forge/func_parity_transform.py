from functools import singledispatch
from typing import Union
import pennylane as qml
from pennylane.operation import Operator
from pennylane.pauli import PauliSentence, PauliWord
from .fermionic import FermiSentence, FermiWord
def parity_transform(fermi_operator: Union[FermiWord, FermiSentence], n: int, ps: bool=False, wire_map: dict=None, tol: float=None) -> Union[Operator, PauliSentence]:
    """Convert a fermionic operator to a qubit operator using the parity mapping.

    .. note::

        Hamiltonians created with this mapping should be used with operators and states that are
        compatible with the parity basis.

    In parity mapping, qubit :math:`j` stores the parity of all :math:`j-1` qubits before it.
    In comparison, :func:`~.jordan_wigner` simply uses qubit :math:`j` to store the occupation number.
    In parity mapping, the fermionic creation and annihilation operators are mapped to the Pauli operators as

    .. math::
        \\begin{align*}
           a^{\\dagger}_0 &= \\left (\\frac{X_0 - iY_0}{2}\xa0 \\right )\\otimes X_1 \\otimes X_2 \\otimes ... X_n, \\\\\\\\
           a^{\\dagger}_n &= \\left (\\frac{Z_{n-1} \\otimes X_n - iY_n}{2} \\right ) \\otimes X_{n+1} \\otimes X_{n+2} \\otimes ... \\otimes X_n
        \\end{align*}

    and

    .. math::
        \\begin{align*}
           a_0 &= \\left (\\frac{X_0 + iY_0}{2}\xa0 \\right )\\otimes X_1 \\otimes X_2 \\otimes ... X_n,\\\\\\\\
           a_n &= \\left (\\frac{Z_{n-1} \\otimes X_n + iY_n}{2} \\right ) \\otimes X_{n+1} \\otimes X_{n+2} \\otimes ... \\otimes X_n
        \\end{align*}

    where :math:`X`, :math:`Y`, and :math:`Z` are the Pauli operators and :math:`n` is the number of qubits, i.e., spin orbitals.

    Args:
        fermi_operator(FermiWord, FermiSentence): the fermionic operator
        n (int): number of qubits, i.e., spin orbitals in the system
        ps (bool): whether to return the result as a :class:`~.PauliSentence` instead of an
            :class:`~.Operator`. Defaults to ``False``.
        wire_map (dict): a dictionary defining how to map the orbitals of
            the Fermi operator to qubit wires. If ``None``, the integers used to
            order the orbitals will be used as wire labels. Defaults to ``None``.
        tol (float): tolerance for discarding the imaginary part of the coefficients

    Returns:
        Union[PauliSentence, Operator]: a linear combination of qubit operators

    **Example**

    >>> w = FermiWord({(0, 0) : '+', (1, 1) : '-'})
    >>> parity_transform(w, n=6)
    (
        -0.25j * Y(0)
    + (-0.25+0j) * (X(0) @ Z(1))
    + (0.25+0j) * X(0)
    + 0.25j * (Y(0) @ Z(1))
    )

    >>> parity_transform(w, n=6, ps=True)
    -0.25j * Y(0)
    + (-0.25+0j) * X(0) @ Z(1)
    + (0.25+0j) * X(0)
    + 0.25j * Y(0) @ Z(1)

    >>> parity_transform(w, n=6, ps=True, wire_map={0: 2, 1: 3})
    -0.25j * Y(2)
    + (-0.25+0j) * X(2) @ Z(3)
    + (0.25+0j) * X(2)
    + 0.25j * Y(2) @ Z(3)
    """
    return _parity_transform_dispatch(fermi_operator, n, ps, wire_map, tol)