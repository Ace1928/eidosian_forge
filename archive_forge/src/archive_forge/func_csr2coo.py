import functools as _functools
import numpy as _numpy
import platform as _platform
import cupy as _cupy
from cupy_backends.cuda.api import driver as _driver
from cupy_backends.cuda.api import runtime as _runtime
from cupy_backends.cuda.libs import cusparse as _cusparse
from cupy._core import _dtype
from cupy.cuda import device as _device
from cupy.cuda import stream as _stream
from cupy import _util
import cupyx.scipy.sparse
def csr2coo(x, data, indices):
    """Converts a CSR-matrix to COO format.

    Args:
        x (cupyx.scipy.sparse.csr_matrix): A matrix to be converted.
        data (cupy.ndarray): A data array for converted data.
        indices (cupy.ndarray): An index array for converted data.

    Returns:
        cupyx.scipy.sparse.coo_matrix: A converted matrix.

    """
    if not check_availability('csr2coo'):
        raise RuntimeError('csr2coo is not available.')
    handle = _device.get_cusparse_handle()
    m = x.shape[0]
    nnz = x.nnz
    row = _cupy.empty(nnz, 'i')
    _cusparse.xcsr2coo(handle, x.indptr.data.ptr, nnz, m, row.data.ptr, _cusparse.CUSPARSE_INDEX_BASE_ZERO)
    return cupyx.scipy.sparse.coo_matrix((data, (row, indices)), shape=x.shape)