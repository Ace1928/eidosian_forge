import numpy as np
from .layer import Layer1Q, Layer2Q
def mul_right_q1(self, layer: Layer1Q, temp_mat: np.ndarray, dagger: bool):
    """
        Multiplies ``NxN`` matrix, wrapped by this object, by a 1-qubit layer
        matrix of the right, where ``N`` is the actual size of matrices involved,
        ``N = 2^{num. of qubits}``.

        Args:
            layer: 1-qubit layer, i.e. the layer with just one non-trivial
                   1-qubit gate and other gates are just identity operators.
            temp_mat: a temporary NxN matrix used as a workspace.
            dagger: if true, the right-hand side matrix will be taken as
                    conjugate transposed.
        """
    gmat, perm, inv_perm = layer.get_attr()
    mat = self._mat
    dim = perm.size
    np.take(mat, np.take(self._right_perm, perm, out=self._temp_perm), axis=1, out=temp_mat)
    gmat_right = np.conj(gmat, out=self._temp_g2x2).T if dagger else gmat
    for i in range(0, dim, 2):
        mat[:, i:i + 2] = np.dot(temp_mat[:, i:i + 2], gmat_right, out=self._temp_slice_dim_x_2)
    self._right_perm[:] = inv_perm