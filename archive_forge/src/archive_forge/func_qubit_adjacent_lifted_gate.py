from typing import List, Sequence, Tuple, Union, cast
import numpy as np
from pyquil.experiment._setting import TensorProductState
from pyquil.paulis import PauliSum, PauliTerm
from pyquil.quil import Program
from pyquil.quilatom import Parameter
from pyquil.quilbase import Gate, Halt, _strip_modifiers
from pyquil.simulation.matrices import SWAP, STATES, QUANTUM_GATES
def qubit_adjacent_lifted_gate(i: int, matrix: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    Lifts input k-qubit gate on adjacent qubits starting from qubit i
    to complete Hilbert space of dimension 2 ** num_qubits.

    Ex: 1-qubit gate, lifts from qubit i
    Ex: 2-qubit gate, lifts from qubits (i+1, i)
    Ex: 3-qubit gate, lifts from qubits (i+2, i+1, i), operating in that order

    In general, this takes a k-qubit gate (2D matrix 2^k x 2^k) and lifts
    it to the complete Hilbert space of dim 2^num_qubits, as defined by
    the right-to-left tensor product (1) in arXiv:1608.03355.

    Developer note: Quil and the QVM like qubits to be ordered such that qubit 0 is on the right.
    Therefore, in ``qubit_adjacent_lifted_gate``, ``lifted_pauli``, and ``lifted_state_operator``,
    we build up the lifted matrix by performing the kronecker product from right to left.

    Note that while the qubits are addressed in decreasing order,
    starting with num_qubit - 1 on the left and ending with qubit 0 on the
    right (in a little-endian fashion), gates are still lifted to apply
    on qubits in increasing index (right-to-left) order.

    :param i: starting qubit to lift matrix from (incr. index order)
    :param matrix: the matrix to be lifted
    :param n_qubits: number of overall qubits present in space

    :return: matrix representation of operator acting on the
        complete Hilbert space of all num_qubits.
    """
    n_rows, n_cols = matrix.shape
    assert n_rows == n_cols, 'Matrix must be square'
    gate_size = np.log2(n_rows)
    assert gate_size == int(gate_size), 'Matrix must be 2^n by 2^n'
    gate_size = int(gate_size)
    bottom_matrix = np.eye(2 ** i, dtype=np.complex128)
    top_qubits = n_qubits - i - gate_size
    top_matrix = np.eye(2 ** top_qubits, dtype=np.complex128)
    return np.kron(top_matrix, np.kron(matrix, bottom_matrix))