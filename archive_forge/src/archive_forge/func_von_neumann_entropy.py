from typing import Optional, TYPE_CHECKING, Tuple
import numpy as np
from cirq import protocols, value, _import
from cirq.qis.states import (
def von_neumann_entropy(state: 'cirq.QUANTUM_STATE_LIKE', qid_shape: Optional[Tuple[int, ...]]=None, validate: bool=True, atol: float=1e-07) -> float:
    """Calculates the von Neumann entropy of a quantum state in bits.

    The Von Neumann entropy is defined as $ - trace( \\rho ln \\rho)$, for
    a density matrix $\\rho$.  This gives the amount of entropy in 'ebits'
    (bits of bipartite entanglement).

    If `state` is a square matrix, it is assumed to be a density matrix rather
    than a (pure) state tensor.

    Args:
        state: The quantum state.
        qid_shape: The qid shape of the given state.
        validate: Whether to check if the given state is a valid quantum state.
        atol: Absolute numerical tolerance to use for validation.

    Returns:
        The calculated von Neumann entropy.

    Raises:
        ValueError: Invalid quantum state.

    References:
        https://en.wikipedia.org/wiki/Von_Neumann_entropy
    """
    if isinstance(state, QuantumState) and state._is_density_matrix():
        state = state.data
    if isinstance(state, np.ndarray) and state.ndim == 2 and (state.shape[0] == state.shape[1]):
        if validate:
            if qid_shape is None:
                qid_shape = (state.shape[0],)
            validate_density_matrix(state, qid_shape=qid_shape, dtype=state.dtype, atol=atol)
        eigenvalues = np.linalg.eigvalsh(state)
        return stats.entropy(np.abs(eigenvalues), base=2)
    if validate:
        _ = quantum_state(state, qid_shape=qid_shape, copy=False, validate=True, atol=atol)
    return 0.0