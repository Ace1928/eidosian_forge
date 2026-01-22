import numpy
import cupy
from cupy import _core
from cupyx import cusparse
from cupyx.scipy.sparse import _base
from cupyx.scipy.sparse import _csc
from cupyx.scipy.sparse import _csr
from cupyx.scipy.sparse import _data as sparse_data
from cupyx.scipy.sparse import _util
from cupyx.scipy.sparse import _sputils
def setdiag(self, values, k=0):
    """Set diagonal or off-diagonal elements of the array.

        Args:
            values (ndarray): New values of the diagonal elements. Values may
                have any length. If the diagonal is longer than values, then
                the remaining diagonal entries will not be set. If values are
                longer than the diagonal, then the remaining values are
                ignored. If a scalar value is given, all of the diagonal is set
                to it.
            k (int, optional): Which off-diagonal to set, corresponding to
                elements a[i,i+k]. Default: 0 (the main diagonal).

        """
    M, N = self.shape
    if k > 0 and k >= N or (k < 0 and -k >= M):
        raise ValueError('k exceeds matrix dimensions')
    if values.ndim and (not len(values)):
        return
    idx_dtype = self.row.dtype
    full_keep = self.col - self.row != k
    if k < 0:
        max_index = min(M + k, N)
        if values.ndim:
            max_index = min(max_index, len(values))
        keep = cupy.logical_or(full_keep, self.col >= max_index)
        new_row = cupy.arange(-k, -k + max_index, dtype=idx_dtype)
        new_col = cupy.arange(max_index, dtype=idx_dtype)
    else:
        max_index = min(M, N - k)
        if values.ndim:
            max_index = min(max_index, len(values))
        keep = cupy.logical_or(full_keep, self.row >= max_index)
        new_row = cupy.arange(max_index, dtype=idx_dtype)
        new_col = cupy.arange(k, k + max_index, dtype=idx_dtype)
    if values.ndim:
        new_data = values[:max_index]
    else:
        new_data = cupy.full(max_index, values, dtype=self.dtype)
    self.row = cupy.concatenate((self.row[keep], new_row))
    self.col = cupy.concatenate((self.col[keep], new_col))
    self.data = cupy.concatenate((self.data[keep], new_data))
    self.has_canonical_format = False