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
def reduce_statevector(state, indices, check_state=False, c_dtype='complex128'):
    """Compute the density matrix from a state vector.

    Args:
        state (tensor_like): 1D or 2D tensor state vector. This tensor should of size ``(2**N,)``
            or ``(batch_dim, 2**N)``, for some integer value ``N``.
        indices (list(int)): List of indices in the considered subsystem.
        check_state (bool): If True, the function will check the state validity (shape and norm).
        c_dtype (str): Complex floating point precision type.

    Returns:
        tensor_like: Density matrix of size ``(2**len(indices), 2**len(indices))`` or ``(batch_dim, 2**len(indices), 2**len(indices))``

    .. seealso:: :func:`pennylane.math.reduce_dm` and :func:`pennylane.density_matrix`

    **Example**

    >>> x = np.array([1, 0, 0, 0])
    >>> reduce_statevector(x, indices=[0])
    [[1.+0.j 0.+0.j]
    [0.+0.j 0.+0.j]]

    >>> y = [1, 0, 1, 0] / np.sqrt(2)
    >>> reduce_statevector(y, indices=[0])
    [[0.5+0.j 0.5+0.j]
     [0.5+0.j 0.5+0.j]]

    >>> reduce_statevector(y, indices=[1])
    [[1.+0.j 0.+0.j]
     [0.+0.j 0.+0.j]]

    >>> z = tf.Variable([1, 0, 0, 0], dtype=tf.complex128)
    >>> reduce_statevector(z, indices=[1])
    tf.Tensor(
    [[1.+0.j 0.+0.j]
     [0.+0.j 0.+0.j]], shape=(2, 2), dtype=complex128)

    >>> x = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    >>> reduce_statevector(x, indices=[1])
    array([[[1.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j]],
           [[0.+0.j, 0.+0.j],
            [0.+0.j, 1.+0.j]]])
    """
    state = cast(state, dtype=c_dtype)
    if check_state:
        _check_state_vector(state)
    if len(np.shape(state)) == 1:
        batch_dim, dim = (None, np.shape(state)[0])
    else:
        batch_dim, dim = np.shape(state)[:2]
        if batch_dim is None:
            batch_dim = -1
    num_wires = int(np.log2(dim))
    consecutive_wires = list(range(num_wires))
    if batch_dim is None:
        state = qml.math.stack([state])
    state = np.reshape(state, [batch_dim if batch_dim is not None else 1] + [2] * num_wires)
    indices1 = ABC[1:num_wires + 1]
    indices2 = ''.join([ABC[num_wires + i + 1] if i in indices else ABC[i + 1] for i in consecutive_wires])
    target = ''.join([ABC[i + 1] for i in sorted(indices)] + [ABC[num_wires + i + 1] for i in sorted(indices)])
    density_matrix = einsum(f'a{indices1},a{indices2}->a{target}', state, np.conj(state), optimize='greedy')
    if batch_dim is None:
        density_matrix = np.reshape(density_matrix, (2 ** len(indices), 2 ** len(indices)))
    else:
        density_matrix = np.reshape(density_matrix, (batch_dim, 2 ** len(indices), 2 ** len(indices)))
    return _permute_dense_matrix(density_matrix, sorted(indices), indices, batch_dim)