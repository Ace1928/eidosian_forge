import warnings
from itertools import product
import numpy as np
import pennylane as qml
from pennylane.operation import Tensor, active_new_opmath
from pennylane.pauli import pauli_sentence
from pennylane.wires import Wires
def import_operator(qubit_observable, format='openfermion', wires=None, tol=10000000000.0):
    """Convert an external operator to a PennyLane operator.

    We currently support `OpenFermion <https://quantumai.google/openfermion>`__ operators: the function accepts most types of
    OpenFermion qubit operators, such as those corresponding to Pauli words and sums of Pauli words.

    Args:
        qubit_observable: external qubit operator that will be converted
        format (str): the format of the operator object to convert from
        wires (.Wires, list, tuple, dict): Custom wire mapping used to convert the external qubit
            operator to a PennyLane operator.
            For types ``Wires``/list/tuple, each item in the iterable represents a wire label
            for the corresponding qubit index.
            For type dict, only int-keyed dictionaries (for qubit-to-wire conversion) are accepted.
            If ``None``, the identity map (e.g., ``0->0, 1->1, ...``) will be used.
        tol (float): Tolerance in `machine epsilon <https://numpy.org/doc/stable/reference/generated/numpy.real_if_close.html>`_
            for the imaginary part of the coefficients in ``qubit_observable``.
            Coefficients with imaginary part less than :math:`(2.22 \\cdot 10^{-16}) \\cdot \\text{tol}` are considered to be real.

    Returns:
        (.Operator): PennyLane operator representing any operator expressed as linear combinations of
        Pauli words, e.g.,
        :math:`\\sum_{k=0}^{N-1} c_k O_k`

    **Example**

    >>> from openfermion import QubitOperator
    >>> h_of = QubitOperator('X0 X1 Y2 Y3', -0.0548) + QubitOperator('Z0 Z1', 0.14297)
    >>> h_pl = import_operator(h_of, format='openfermion')
    >>> print(h_pl)
    (0.14297) [Z0 Z1]
    + (-0.0548) [X0 X1 Y2 Y3]

    If the new op-math is active, an arithmetic operator is returned instead.

    >>> qml.operation.enable_new_opmath()
    >>> h_pl = import_operator(h_of, format='openfermion')
    >>> print(h_pl)
    (-0.0548 * X(0 @ X(1) @ Y(2) @ Y(3))) + (0.14297 * Z(0 @ Z(1)))
    """
    if format not in ['openfermion']:
        raise TypeError(f'Converter does not exist for {format} format.')
    coeffs = np.array([np.real_if_close(coef, tol=tol) for coef in qubit_observable.terms.values()])
    if any(np.iscomplex(coeffs)):
        warnings.warn(f'The coefficients entering the QubitOperator must be real; got complex coefficients in the operator {list(coeffs[np.iscomplex(coeffs)])}')
    if active_new_opmath():
        return qml.dot(*_openfermion_to_pennylane(qubit_observable, wires=wires))
    return qml.Hamiltonian(*_openfermion_to_pennylane(qubit_observable, wires=wires))