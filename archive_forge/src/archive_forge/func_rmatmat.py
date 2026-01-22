import warnings
import cupy
from cupyx.scipy import sparse
from cupyx.scipy.sparse import _util
def rmatmat(self, X):
    """Adjoint matrix-matrix multiplication.
        """
    if X.ndim != 2:
        raise ValueError('expected 2-d ndarray or matrix, not %d-d' % X.ndim)
    if X.shape[0] != self.shape[0]:
        raise ValueError('dimension mismatch: %r, %r' % (self.shape, X.shape))
    Y = self._rmatmat(X)
    return Y