import numpy
import cupy
from cupy import _core
from cupyx.scipy.sparse import _util
from cupyx.scipy.sparse import _sputils
def todok(self, copy=False):
    """Convert this matrix to Dictionary Of Keys format."""
    return self.tocsr(copy=copy).todok(copy=False)