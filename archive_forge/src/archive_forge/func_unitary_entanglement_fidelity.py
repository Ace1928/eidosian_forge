import itertools
from typing import Union, Sequence, Optional
import numpy as np
from cirq.value import random_state
def unitary_entanglement_fidelity(U_actual: np.ndarray, U_ideal: np.ndarray) -> np.ndarray:
    """Entanglement fidelity between two unitaries.

    For unitary matrices, this is related to the average unitary fidelity F
    as

    :math:`F = \\frac{F_e d + 1}{d + 1}`
    where d is the matrix dimension.

    Args:
        U_actual : Matrix whose fidelity to U_ideal will be computed. This may
            be a non-unitary matrix, i.e. the projection of a larger unitary
            matrix into the computational subspace.
        U_ideal : Unitary matrix to which U_actual will be compared.

    Both arguments may be vectorized, in that their shapes may be of the form
    (...,M,M) (as long as both shapes can be broadcast together).

    Returns:
        The entanglement fidelity between the two unitaries. For inputs with
        shape (...,M,M), the output has shape (...).

    """
    U_actual = np.asarray(U_actual)
    U_ideal = np.asarray(U_ideal)
    assert U_actual.shape[-1] == U_actual.shape[-2], "Inputs' trailing dimensions must be equal (square)."
    dim = U_ideal.shape[-1]
    prod_trace = np.einsum('...ba,...ba->...', U_actual.conj(), U_ideal)
    return np.real(np.abs(prod_trace) / dim) ** 2