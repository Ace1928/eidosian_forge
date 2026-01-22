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
def partial_trace(matrix, indices, c_dtype='complex128'):
    """Compute the reduced density matrix by tracing out the provided indices.

    Args:
        matrix (tensor_like): 2D or 3D density matrix tensor. For a 2D tensor, the size is assumed to be
            ``(2**n, 2**n)``, for some integer number of wires ``n``. For a 3D tensor, the first dimension is assumed to be the batch dimension, ``(batch_dim, 2**N, 2**N)``.

        indices (list(int)): List of indices to be traced.

    Returns:
        tensor_like: (reduced) Density matrix of size ``(2**len(wires), 2**len(wires))``

    .. seealso:: :func:`pennylane.math.reduce_dm`, and :func:`pennylane.math.reduce_statevector`

    **Example**

    We can compute the partial trace of the matrix ``x`` with respect to its 0th index.

    >>> x = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    >>> partial_trace(x, indices=[0])
    array([[1, 0], [0, 0]])

    We can also pass a batch of matrices ``x`` to the function and return the partial trace of each matrix with respect to each matrix's 0th index.

    >>> x = np.array([
        [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    ])
    >>> partial_trace(x, indices=[0])
    array([[[1, 0], [0, 0]], [[0, 0], [0, 1]]])

    The partial trace can also be computed with respect to multiple indices within different frameworks such as TensorFlow.

    >>> x = tf.Variable([[[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    ... [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]], dtype=tf.complex128)
    >>> partial_trace(x, indices=[1])
    <tf.Tensor: shape=(2, 2, 2), dtype=complex128, numpy=
    array([[[1.+0.j, 0.+0.j], [0.+0.j, 0.+0.j]], [[1.+0.j, 0.+0.j], [0.+0.j, 0.+0.j]]])>

    """
    matrix = cast(matrix, dtype=c_dtype)
    if qml.math.ndim(matrix) == 2:
        is_batched = False
        batch_dim, dim = (1, matrix.shape[1])
    else:
        is_batched = True
        batch_dim, dim = matrix.shape[:2]
    if get_interface(matrix) in ['autograd', 'tensorflow']:
        return _batched_partial_trace_nonrep_indices(matrix, is_batched, indices, batch_dim, dim)
    num_indices = int(np.log2(dim))
    rho_dim = 2 * num_indices
    matrix = np.reshape(matrix, [batch_dim] + [2] * 2 * num_indices)
    indices = np.sort(indices)
    for i, target_index in enumerate(indices):
        target_index = target_index - i
        state_indices = ABC[1:rho_dim - 2 * i + 1]
        state_indices = list(state_indices)
        target_letter = state_indices[target_index]
        state_indices[target_index + num_indices - i] = target_letter
        state_indices = ''.join(state_indices)
        einsum_indices = f'a{state_indices}'
        matrix = einsum(einsum_indices, matrix)
    number_wires_sub = num_indices - len(indices)
    reduced_density_matrix = np.reshape(matrix, (batch_dim, 2 ** number_wires_sub, 2 ** number_wires_sub))
    return reduced_density_matrix if is_batched else reduced_density_matrix[0]