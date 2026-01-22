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
def eliminate_zeros(self):
    """Removes zero entories in place."""
    ind = self.data != 0
    self.data = self.data[ind]
    self.row = self.row[ind]
    self.col = self.col[ind]