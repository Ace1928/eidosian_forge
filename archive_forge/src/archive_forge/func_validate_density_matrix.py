from typing import Any, cast, Iterable, List, Optional, Sequence, Set, TYPE_CHECKING, Tuple, Union
import itertools
import numpy as np
from cirq import value
from cirq._doc import document
def validate_density_matrix(density_matrix: np.ndarray, *, qid_shape: Tuple[int, ...], dtype: Optional['DTypeLike']=None, atol: float=1e-07) -> None:
    """Checks that the given density matrix is valid.

    Args:
        density_matrix: The density matrix to validate.
        qid_shape: The expected qid shape.
        dtype: The expected dtype.
        atol: Absolute numerical tolerance.

    Raises:
        ValueError: The density matrix does not have the correct dtype.
        ValueError: The density matrix does not have the correct shape.
            It should be a square matrix with dimension prod(qid_shape).
        ValueError: The density matrix is not Hermitian.
        ValueError: The density matrix does not have trace 1.
        ValueError: The density matrix is not positive semidefinite.
    """
    if dtype and density_matrix.dtype != dtype:
        raise ValueError(f'Incorrect dtype for density matrix: Expected {dtype} but has dtype {density_matrix.dtype}.')
    expected_shape = (np.prod(qid_shape, dtype=np.int64),) * 2
    if density_matrix.shape != expected_shape:
        raise ValueError(f'Incorrect shape for density matrix: Expected {expected_shape} but has shape {density_matrix.shape}.')
    if not np.allclose(density_matrix, density_matrix.conj().T, atol=atol):
        raise ValueError('The density matrix is not hermitian.')
    trace = np.trace(density_matrix)
    if not np.isclose(trace, 1.0, atol=atol):
        raise ValueError(f'Density matrix does not have trace 1. Instead, it has trace {trace}.')
    if not np.all(np.linalg.eigvalsh(density_matrix) > -atol):
        raise ValueError('The density matrix is not positive semidefinite.')