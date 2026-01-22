import functools
import itertools
import numpy as np
import scipy
import pennylane as qml
from pennylane.operation import active_new_opmath
from pennylane.pauli import PauliSentence, PauliWord, pauli_sentence, simplify
from pennylane.pauli.utils import _binary_matrix_from_pws
from pennylane.wires import Wires
def paulix_ops(generators, num_qubits):
    """Generate the single qubit Pauli-X operators :math:`\\sigma^{x}_{i}` for each symmetry :math:`\\tau_j`,
    such that it anti-commutes with :math:`\\tau_j` and commutes with all others symmetries :math:`\\tau_{k\\neq j}`.
    These are required to obtain the Clifford operators :math:`U` for the Hamiltonian :math:`H`.

    Args:
        generators (list[Operator]): list of generators of symmetries, :math:`\\tau`'s,
            for the Hamiltonian
        num_qubits (int): number of wires required to define the Hamiltonian

    Return:
        list[Observable]: list of single-qubit Pauli-X operators which will be used to build the
        Clifford operators :math:`U`.

    **Example**

    >>> generators = [qml.Hamiltonian([1.0], [qml.Z(0) @ qml.Z(1)]),
    ...               qml.Hamiltonian([1.0], [qml.Z(0) @ qml.Z(2)]),
    ...               qml.Hamiltonian([1.0], [qml.Z(0) @ qml.Z(3)])]
    >>> paulix_ops(generators, 4)
    [X(1), X(2), X(3)]
    """
    ops_generator = functools.reduce(lambda a, b: list(a) + list(b), [pauli_sentence(g) for g in generators])
    bmat = _binary_matrix_from_pws(ops_generator, num_qubits)
    paulixops = []
    for row in range(bmat.shape[0]):
        bmatrow = bmat[row]
        bmatrest = np.delete(bmat, row, axis=0)
        for col in range(bmat.shape[1] // 2)[::-1]:
            if bmatrow[col] and np.array_equal(bmatrest[:, col], np.zeros(bmat.shape[0] - 1, dtype=int)):
                paulixops.append(qml.X(col))
                break
    return paulixops