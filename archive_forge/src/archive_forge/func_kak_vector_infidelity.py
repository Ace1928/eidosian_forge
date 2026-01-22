import itertools
from typing import Union, Sequence, Optional
import numpy as np
from cirq.value import random_state
def kak_vector_infidelity(k_vec_a: np.ndarray, k_vec_b: np.ndarray, ignore_equivalent_vectors: bool=False) -> np.ndarray:
    """The locally invariant infidelity between two KAK vectors.

    This is the quantity

    $$
    \\min 1 - F_e( \\exp(i k_a · (XX,YY,ZZ)) kL \\exp(i k_b · (XX,YY,ZZ)) kR)
    $$

    where $F_e$ is the entanglement (process) fidelity and the minimum is taken
    over all 1-local unitaries kL, kR.

    Args:
        k_vec_a: A 3-vector or tensor of 3-vectors with shape (...,3).
        k_vec_b: A 3-vector or tensor of 3-vectors with shape (...,3). If both
            k_vec_a and k_vec_b are tensors, their shapes must be compatible
            for broadcasting.
        ignore_equivalent_vectors: If True, the calculation ignores any other
            KAK vectors that are equivalent to the inputs under local unitaries.
            The resulting infidelity is then only an upper bound to the true
            infidelity.

    Returns:
        An ndarray storing the locally invariant infidelity between the inputs.
        If k_vec_a or k_vec_b is a tensor, the result is vectorized.
    """
    k_vec_a, k_vec_b = (np.asarray(k_vec_a), np.asarray(k_vec_b))
    if ignore_equivalent_vectors:
        k_diff = k_vec_a - k_vec_b
        out = 1 - np.prod(np.cos(k_diff), axis=-1) ** 2
        out -= np.prod(np.sin(k_diff), axis=-1) ** 2
        return out
    if k_vec_a.size < k_vec_b.size:
        k_vec_a, k_vec_b = (k_vec_b, k_vec_a)
    k_vec_a = k_vec_a[..., np.newaxis, :]
    k_vec_b = _kak_equivalent_vectors(k_vec_b)
    k_diff = k_vec_a - k_vec_b
    out = 1 - np.prod(np.cos(k_diff), axis=-1) ** 2
    out -= np.prod(np.sin(k_diff), axis=-1) ** 2
    return out.min(axis=-1)