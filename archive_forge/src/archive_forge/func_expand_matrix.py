import itertools
from functools import reduce
from typing import Generator, Iterable, Tuple
import numpy as np
from scipy.sparse import csr_matrix, eye, kron
import pennylane as qml
from pennylane.wires import Wires
def expand_matrix(mat, wires, wire_order=None, sparse_format='csr'):
    """Re-express a matrix acting on a subspace defined by a set of wire labels
    according to a global wire order.

    Args:
        mat (tensor_like): matrix to expand
        wires (Iterable): wires determining the subspace that ``mat`` acts on; a matrix of
            dimension :math:`2^n` acts on a subspace of :math:`n` wires
        wire_order (Iterable): global wire order, which has to contain all wire labels in ``wires``, but can also
            contain additional labels
        sparse_format (str): if ``mat`` is a SciPy sparse matrix then this is the string representing the
            preferred scipy sparse matrix format to cast the expanded matrix too

    Returns:
        tensor_like: expanded matrix

    **Example**

    If the wire order is ``None`` or identical to ``wires``, the original matrix gets returned:

    >>> matrix = np.array([[1, 2, 3, 4],
    ...                    [5, 6, 7, 8],
    ...                    [9, 10, 11, 12],
    ...                    [13, 14, 15, 16]])
    >>> print(expand_matrix(matrix, wires=[0, 2], wire_order=[0, 2]))
    [[ 1  2  3  4]
     [ 5  6  7  8]
     [ 9 10 11 12]
     [13 14 15 16]]
    >>> print(expand_matrix(matrix, wires=[0, 2]))
    [[ 1  2  3  4]
     [ 5  6  7  8]
     [ 9 10 11 12]
     [13 14 15 16]]

    If the wire order is a permutation of ``wires``, the entries of the matrix get permuted:

    >>> print(expand_matrix(matrix, wires=[0, 2], wire_order=[2, 0]))
    [[ 1  3  2  4]
     [ 9 11 10 12]
     [ 5  7  6  8]
     [13 15 14 16]]

    If the wire order contains wire labels not found in ``wires``, the matrix gets expanded:

    >>> print(expand_matrix(matrix, wires=[0, 2], wire_order=[0, 1, 2]))
    [[ 1  2  0  0  3  4  0  0]
     [ 5  6  0  0  7  8  0  0]
     [ 0  0  1  2  0  0  3  4]
     [ 0  0  5  6  0  0  7  8]
     [ 9 10  0  0 11 12  0  0]
     [13 14  0  0 15 16  0  0]
     [ 0  0  9 10  0  0 11 12]
     [ 0  0 13 14  0  0 15 16]]

    The method works with tensors from all autodifferentiation frameworks, for example:

    >>> matrix_torch = torch.tensor([[1., 2.],
    ...                              [3., 4.]], requires_grad=True)
    >>> res = expand_matrix(matrix_torch, wires=["b"], wire_order=["a", "b"])
    >>> type(res)
    torch.Tensor
    >>> res.requires_grad
    True

    The method works with scipy sparse matrices, for example:

    >>> from scipy import sparse
    >>> mat = sparse.csr_matrix([[0, 1], [1, 0]])
    >>> qml.math.expand_matrix(mat, wires=[1], wire_order=[0,1]).toarray()
    array([[0., 1., 0., 0.],
           [1., 0., 0., 0.],
           [0., 0., 0., 1.],
           [0., 0., 1., 0.]])

    """
    if wire_order is None or wire_order == wires:
        return mat
    wires = list(wires)
    wire_order = list(wire_order)
    interface = qml.math.get_interface(mat)
    shape = qml.math.shape(mat)
    batch_dim = shape[0] if len(shape) == 3 else None

    def eye_interface(dim):
        if interface == 'scipy':
            return eye(2 ** dim, format='coo')
        return qml.math.cast_like(qml.math.eye(2 ** dim, like=interface), mat)

    def kron_interface(mat1, mat2):
        if interface == 'scipy':
            res = kron(mat1, mat2, format='coo')
            res.eliminate_zeros()
            return res
        if interface == 'torch':
            mat1 = mat1.contiguous()
            mat2 = mat2.contiguous()
        return qml.math.kron(mat1, mat2, like=interface)
    wire_indices = [wire_order.index(wire) for wire in wires]
    subset_wire_order = wire_order[min(wire_indices):max(wire_indices) + 1]
    wire_difference = list(set(subset_wire_order) - set(wires))
    expanded_wires = wires + wire_difference
    if wire_difference:
        if batch_dim is not None:
            batch_matrices = [kron_interface(batch, eye_interface(len(wire_difference))) for batch in mat]
            mat = qml.math.stack(batch_matrices, like=interface)
        else:
            mat = kron_interface(mat, eye_interface(len(wire_difference)))
    if interface == 'scipy':
        mat = _permute_sparse_matrix(mat, expanded_wires, subset_wire_order)
    else:
        mat = _permute_dense_matrix(mat, expanded_wires, subset_wire_order, batch_dim)
    if len(expanded_wires) < len(wire_order):
        mats = []
        num_pre_identities = min(wire_indices)
        if num_pre_identities > 0:
            mats.append((eye_interface(num_pre_identities),))
        mats.append(tuple(mat) if batch_dim else (mat,))
        num_post_identities = len(wire_order) - max(wire_indices) - 1
        if num_post_identities > 0:
            mats.append((eye_interface(num_post_identities),))
        mats_list = list(itertools.product(*mats))
        expanded_batch_matrices = [reduce(kron_interface, mats) for mats in mats_list]
        mat = qml.math.stack(expanded_batch_matrices, like=interface) if len(expanded_batch_matrices) > 1 else expanded_batch_matrices[0]
    return mat.asformat(sparse_format) if interface == 'scipy' else mat