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
def getnnz(self, axis=None):
    """Returns the number of stored values, including explicit zeros."""
    if axis is None:
        return self.data.size
    else:
        raise ValueError