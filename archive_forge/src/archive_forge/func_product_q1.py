import numpy as np
from .layer import Layer1Q, Layer2Q
def product_q1(self, layer: Layer1Q, tmp1: np.ndarray, tmp2: np.ndarray) -> np.complex128:
    """
        Computes and returns: ``Trace(mat @ C) = Trace(mat @ P^T @ gmat @ P) =
        Trace((P @ mat @ P^T) @ gmat) = Trace(C @ (P @ mat @ P^T)) =
        vec(gmat^T)^T @ vec(P @ mat @ P^T)``, where mat is ``NxN`` matrix wrapped
        by this object, ``C`` is matrix representation of the layer ``L``, and gmat
        is 2x2 matrix of underlying 1-qubit gate.

        **Note**: matrix of this class must be finalized beforehand.

        Args:
            layer: 1-qubit layer.
            tmp1: temporary, external matrix used as a workspace.
            tmp2: temporary, external matrix used as a workspace.

        Returns:
            trace of the matrix product.
        """
    mat = self._mat
    gmat, perm, _ = layer.get_attr()
    np.take(np.take(mat, perm, axis=0, out=tmp1), perm, axis=1, out=tmp2)
    gmat_t, tmp3 = (self._temp_g2x2, self._temp_2x2)
    np.copyto(gmat_t, gmat.T)
    _sum = 0.0
    for i in range(0, mat.shape[0], 2):
        tmp3[:, :] = tmp2[i:i + 2, i:i + 2]
        _sum += np.dot(gmat_t.ravel(), tmp3.ravel())
    return np.complex128(_sum)