from typing import Any, cast, Iterable, List, Optional, Sequence, Set, TYPE_CHECKING, Tuple, Union
import itertools
import numpy as np
from cirq import value
from cirq._doc import document
def density_matrix_from_state_vector(state_vector: np.ndarray, indices: Optional[Iterable[int]]=None, qid_shape: Optional[Tuple[int, ...]]=None) -> np.ndarray:
    """Returns the density matrix of the state vector.

    Calculate the density matrix for the system on the given qubit indices,
    with the qubits not in indices that are present in state vector traced out.
    If indices is None the full density matrix for `state_vector` is returned.
    We assume `state_vector` follows the standard Kronecker convention of
    numpy.kron (big-endian).

    For example:
    state_vector = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=np.complex64)
    indices = None
    gives us

        $$
        \\rho = \\begin{bmatrix}
                0.5 & 0.5 \\\\
                0.5 & 0.5
        \\end{bmatrix}
        $$

    Args:
        state_vector: A sequence representing a state vector in which
            the ordering mapping to qubits follows the standard Kronecker
            convention of numpy.kron (big-endian).
        indices: list containing indices for qubits that you would like
            to include in the density matrix (i.e.) qubits that WON'T
            be traced out. follows the standard Kronecker convention of
            numpy.kron.
        qid_shape: specifies the dimensions of the qudits for the input
            `state_vector`.  If not specified, qubits are assumed and the
            `state_vector` must have a dimension a power of two.

    Returns:
        A numpy array representing the density matrix.

    Raises:
        ValueError: if the size of `state_vector` is not a power of 2 and the
            shape is not given or if the shape is given and `state_vector`
            has a size that contradicts this shape.
        IndexError: if the indices are out of range for the number of qubits
            corresponding to `state_vector`.
    """
    shape = validate_qid_shape(state_vector, qid_shape)
    n_qubits = len(shape)
    if indices is None:
        return np.outer(state_vector, np.conj(state_vector))
    indices = list(indices)
    validate_indices(n_qubits, indices)
    state_vector = np.asarray(state_vector).reshape(shape)
    sum_inds = np.array(range(n_qubits))
    sum_inds[indices] += n_qubits
    rho = np.einsum(state_vector, list(range(n_qubits)), np.conj(state_vector), cast(List, sum_inds.tolist()), indices + cast(List, sum_inds[indices].tolist()))
    new_shape = np.prod([shape[i] for i in indices], dtype=np.int64)
    return rho.reshape((new_shape, new_shape))