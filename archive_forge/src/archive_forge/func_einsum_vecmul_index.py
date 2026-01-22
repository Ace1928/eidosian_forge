from __future__ import annotations
from string import ascii_uppercase, ascii_lowercase
import numpy as np
import qiskit.circuit.library.standard_gates as gates
from qiskit.exceptions import QiskitError
def einsum_vecmul_index(gate_indices: list[int], number_of_qubits: int) -> str:
    """Return the index string for Numpy.einsum matrix-vector multiplication.

    The returned indices are to perform a matrix multiplication A.v where
    the matrix A is an M-qubit matrix, vector v is an N-qubit vector, and
    M <= N, and identity matrices are implied on the subsystems where A has no
    support on v.

    Args:
        gate_indices (list[int]): the indices of the right matrix subsystems
                                  to contract with the left matrix.
        number_of_qubits (int): the total number of qubits for the right matrix.

    Returns:
        str: An indices string for the Numpy.einsum function.
    """
    mat_l, mat_r, tens_lin, tens_lout = _einsum_matmul_index_helper(gate_indices, number_of_qubits)
    return f'{mat_l}{mat_r}, {tens_lin}->{tens_lout}'