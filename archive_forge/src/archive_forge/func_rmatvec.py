import warnings
import cupy
from cupyx.scipy import sparse
from cupyx.scipy.sparse import _util
def rmatvec(self, x):
    """Adjoint matrix-vector multiplication.
        """
    M, N = self.shape
    if x.shape != (M,) and x.shape != (M, 1):
        raise ValueError('dimension mismatch')
    y = self._rmatvec(x)
    if x.ndim == 1:
        y = y.reshape(N)
    elif x.ndim == 2:
        y = y.reshape(N, 1)
    else:
        raise ValueError('invalid shape returned by user-defined rmatvec()')
    return y