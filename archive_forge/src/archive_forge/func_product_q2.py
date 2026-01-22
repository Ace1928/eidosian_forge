import numpy as np
from .layer import Layer1Q, Layer2Q
def product_q2(self, layer: Layer2Q, tmp1: np.ndarray, tmp2: np.ndarray) -> np.complex128:
    """
        Computes and returns: ``Trace(mat @ C) = Trace(mat @ P^T @ gmat @ P) =
        Trace((P @ mat @ P^T) @ gmat) = Trace(C @ (P @ mat @ P^T)) =
        vec(gmat^T)^T @ vec(P @ mat @ P^T)``, where mat is ``NxN`` matrix wrapped
        by this object, ``C`` is matrix representation of the layer ``L``, and gmat
        is 4x4 matrix of underlying 2-qubit gate.

        **Note**: matrix of this class must be finalized beforehand.

        Args:
            layer: 2-qubit layer.
            tmp1: temporary, external matrix used as a workspace.
            tmp2: temporary, external matrix used as a workspace.

        Returns:
            trace of the matrix product.
        """
    mat = self._mat
    gmat, perm, _ = layer.get_attr()
    np.take(np.take(mat, perm, axis=0, out=tmp1), perm, axis=1, out=tmp2)
    bldia = self._temp_block_diag
    np.take(tmp2.ravel(), self._idx_mat.ravel(), axis=0, out=bldia.ravel())
    bldia *= gmat.reshape(-1, gmat.size)
    return np.complex128(np.sum(bldia))