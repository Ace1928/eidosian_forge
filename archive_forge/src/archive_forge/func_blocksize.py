from warnings import warn
import numpy as np
from ._matrix import spmatrix
from ._data import _data_matrix, _minmax_mixin
from ._compressed import _cs_matrix
from ._base import issparse, _formats, _spbase, sparray
from ._sputils import (isshape, getdtype, getdata, to_native, upcast,
from . import _sparsetools
from ._sparsetools import (bsr_matvec, bsr_matvecs, csr_matmat_maxnnz,
@property
def blocksize(self) -> tuple:
    """Block size of the matrix."""
    return self.data.shape[1:]