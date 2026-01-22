import numpy
import cupy
from cupy import _core
from cupyx.scipy.sparse import _util
from cupyx.scipy.sparse import _sputils
def tobsr(self, blocksize=None, copy=False):
    """Convert this matrix to Block Sparse Row format."""
    return self.tocsr(copy=copy).tobsr(copy=False)