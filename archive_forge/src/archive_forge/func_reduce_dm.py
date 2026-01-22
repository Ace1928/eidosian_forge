import functools
import itertools
from string import ascii_letters as ABC
from autoray import numpy as np
from numpy import float64
import pennylane as qml
from . import single_dispatch  # pylint:disable=unused-import
from .matrix_manipulation import _permute_dense_matrix
from .multi_dispatch import diag, dot, scatter_element_add, einsum, get_interface
from .utils import is_abstract, allclose, cast, convert_like, cast_like
def reduce_dm(density_matrix, indices, check_state=False, c_dtype='complex128'):
    """Compute the density matrix from a state represented with a density matrix.

    Args:
        density_matrix (tensor_like): 2D or 3D density matrix tensor. This tensor should be of size ``(2**N, 2**N)`` or
            ``(batch_dim, 2**N, 2**N)``, for some integer number of wires``N``.
        indices (list(int)): List of indices in the considered subsystem.
        check_state (bool): If True, the function will check the state validity (shape and norm).
        c_dtype (str): Complex floating point precision type.

    Returns:
        tensor_like: Density matrix of size ``(2**len(indices), 2**len(indices))`` or ``(batch_dim, 2**len(indices), 2**len(indices))``

    .. seealso:: :func:`pennylane.math.reduce_statevector`, and :func:`pennylane.density_matrix`

    **Example**

    >>> x = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    >>> reduce_dm(x, indices=[0])
    [[1.+0.j 0.+0.j]
     [0.+0.j 0.+0.j]]

    >>> y = [[0.5, 0, 0.5, 0], [0, 0, 0, 0], [0.5, 0, 0.5, 0], [0, 0, 0, 0]]
    >>> reduce_dm(y, indices=[0])
    [[0.5+0.j 0.5+0.j]
     [0.5+0.j 0.5+0.j]]

    >>> reduce_dm(y, indices=[1])
    [[1.+0.j 0.+0.j]
     [0.+0.j 0.+0.j]]

    >>> z = tf.Variable([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=tf.complex128)
    >>> reduce_dm(x, indices=[1])
    tf.Tensor(
    [[1.+0.j 0.+0.j]
     [0.+0.j 0.+0.j]], shape=(2, 2), dtype=complex128)

    >>> x = np.array([[[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    ...               [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]])
    >>> reduce_dm(x, indices=[1])
    array([[[1.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j]],
           [[0.+0.j, 0.+0.j],
            [0.+0.j, 1.+0.j]]])
    """
    density_matrix = cast(density_matrix, dtype=c_dtype)
    if check_state:
        _check_density_matrix(density_matrix)
    if len(np.shape(density_matrix)) == 2:
        batch_dim, dim = (None, density_matrix.shape[0])
    else:
        batch_dim, dim = density_matrix.shape[:2]
    num_indices = int(np.log2(dim))
    consecutive_indices = list(range(num_indices))
    if len(indices) == num_indices:
        return _permute_dense_matrix(density_matrix, consecutive_indices, indices, batch_dim)
    if batch_dim is None:
        density_matrix = qml.math.stack([density_matrix])
    traced_wires = [x for x in consecutive_indices if x not in indices]
    density_matrix = partial_trace(density_matrix, traced_wires, c_dtype=c_dtype)
    if batch_dim is None:
        density_matrix = density_matrix[0]
    return _permute_dense_matrix(density_matrix, sorted(indices), indices, batch_dim)