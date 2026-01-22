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
def csrilu02(a, level_info=False):
    """Computes incomplete LU decomposition for a sparse square matrix.

    Args:
        a (cupyx.scipy.sparse.csr_matrix):
            Sparse matrix with dimension ``(M, M)``.
        level_info (bool):
            True: solves it with level infromation.
            False: solves it without level information.

    Note: ``a`` will be overwritten. This function does not support fill-in
        (only ILU(0) is supported) nor pivoting.
    """
    if not check_availability('csrilu02'):
        raise RuntimeError('csrilu02 is not available.')
    if not cupyx.scipy.sparse.isspmatrix_csr(a):
        raise TypeError('a must be CSR sparse matrix')
    if a.shape[0] != a.shape[1]:
        raise ValueError('invalid shape (a.shape: {})'.format(a.shape))
    if level_info is False:
        policy = _cusparse.CUSPARSE_SOLVE_POLICY_NO_LEVEL
    elif level_info is True:
        policy = _cusparse.CUSPARSE_SOLVE_POLICY_USE_LEVEL
    else:
        raise ValueError('Unknown level_info (actual: {})'.format(level_info))
    dtype = a.dtype
    if dtype.char == 'f':
        t = 's'
    elif dtype.char == 'd':
        t = 'd'
    elif dtype.char == 'F':
        t = 'c'
    elif dtype.char == 'D':
        t = 'z'
    else:
        raise TypeError('Invalid dtype (actual: {})'.format(dtype))
    helper = getattr(_cusparse, t + 'csrilu02_bufferSize')
    analysis = getattr(_cusparse, t + 'csrilu02_analysis')
    solve = getattr(_cusparse, t + 'csrilu02')
    check = getattr(_cusparse, 'xcsrilu02_zeroPivot')
    handle = _device.get_cusparse_handle()
    m = a.shape[0]
    nnz = a.nnz
    desc = MatDescriptor.create()
    desc.set_mat_type(_cusparse.CUSPARSE_MATRIX_TYPE_GENERAL)
    desc.set_mat_index_base(_cusparse.CUSPARSE_INDEX_BASE_ZERO)
    info = _cusparse.createCsrilu02Info()
    ws_size = helper(handle, m, nnz, desc.descriptor, a.data.data.ptr, a.indptr.data.ptr, a.indices.data.ptr, info)
    ws = _cupy.empty((ws_size,), dtype=_numpy.int8)
    position = _numpy.empty((1,), dtype=_numpy.int32)
    analysis(handle, m, nnz, desc.descriptor, a.data.data.ptr, a.indptr.data.ptr, a.indices.data.ptr, info, policy, ws.data.ptr)
    try:
        check(handle, info, position.ctypes.data)
    except Exception:
        raise ValueError('a({0},{0}) is missing'.format(position[0]))
    solve(handle, m, nnz, desc.descriptor, a.data.data.ptr, a.indptr.data.ptr, a.indices.data.ptr, info, policy, ws.data.ptr)
    try:
        check(handle, info, position.ctypes.data)
    except Exception:
        raise ValueError('u({0},{0}) is zero'.format(position[0]))