from typing import Optional, TYPE_CHECKING
import numpy as np
from cirq import linalg, value
def random_orthogonal(dim: int, *, random_state: 'cirq.RANDOM_STATE_OR_SEED_LIKE'=None) -> np.ndarray:
    """Returns a random orthogonal matrix distributed with Haar measure.

    Args:
        dim: The width and height of the matrix.
        random_state: A seed (int) or `np.random.RandomState` class to use when
            generating random values. If not set, defaults to using the module
            methods in `np.random`.

    Returns:
        The sampled orthogonal matrix.

    References:
        'How to generate random matrices from the classical compact groups'
        http://arxiv.org/abs/math-ph/0609050
    """
    random_state = value.parse_random_state(random_state)
    m = random_state.randn(dim, dim)
    q, r = np.linalg.qr(m)
    d = np.diag(r)
    return q * (d / abs(d))