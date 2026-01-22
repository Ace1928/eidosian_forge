from typing import Optional, TYPE_CHECKING
import numpy as np
from cirq import linalg, value
def random_density_matrix(dim: int, *, random_state: 'cirq.RANDOM_STATE_OR_SEED_LIKE'=None) -> np.ndarray:
    """Returns a random density matrix distributed with Hilbert-Schmidt measure.

    Args:
        dim: The width and height of the matrix.
        random_state: A seed to use for random number generation.

    Returns:
        The sampled density matrix.

    Reference:
        'Random Bures mixed states and the distribution of their purity'
        https://arxiv.org/abs/0909.5094
    """
    random_state = value.parse_random_state(random_state)
    mat = random_state.randn(dim, dim) + 1j * random_state.randn(dim, dim)
    mat = mat @ mat.T.conj()
    return mat / np.trace(mat)