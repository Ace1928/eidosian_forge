import itertools
from typing import Union, Sequence, Optional
import numpy as np
from cirq.value import random_state
def kak_vector_to_unitary(vector: np.ndarray) -> np.ndarray:
    """Convert a KAK vector to its unitary matrix equivalent.

    Args:
        vector: A KAK vector shape (..., 3). (Input may be vectorized).

    Returns:
        unitary: Corresponding 2-qubit unitary, of the form
           $exp( i k_x \\sigma_x \\sigma_x + i k_y \\sigma_y \\sigma_y
                + i k_z \\sigma_z \\sigma_z)$.
           matrix or tensor of matrices of shape (..., 4,4).
    """
    vector = np.asarray(vector)
    gens = np.einsum('...a,abc->...bc', vector, _kak_gens)
    evals, evecs = np.linalg.eigh(gens)
    return np.einsum('...ab,...b,...cb', evecs, np.exp(1j * evals), evecs.conj())