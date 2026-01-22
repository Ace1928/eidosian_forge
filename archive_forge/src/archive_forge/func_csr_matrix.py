import ctypes
import warnings
import operator
from array import array as native_array
import numpy as np
from ..base import NotSupportedForSparseNDArray
from ..base import _LIB, numeric_types
from ..base import c_array_buf, mx_real_t, integer_types
from ..base import NDArrayHandle, check_call
from ..context import Context, current_context
from . import _internal
from . import op
from ._internal import _set_ndarray_class
from .ndarray import NDArray, _storage_type, _DTYPE_NP_TO_MX, _DTYPE_MX_TO_NP
from .ndarray import _STORAGE_TYPE_STR_TO_ID, _STORAGE_TYPE_ROW_SPARSE, _STORAGE_TYPE_CSR, _int64_enabled
from .ndarray import _STORAGE_TYPE_UNDEFINED, _STORAGE_TYPE_DEFAULT
from .ndarray import zeros as _zeros_ndarray
from .ndarray import array as _array
from .ndarray import _ufunc_helper
def csr_matrix(arg1, shape=None, ctx=None, dtype=None):
    """Creates a `CSRNDArray`, an 2D array with compressed sparse row (CSR) format.

    The CSRNDArray can be instantiated in several ways:

    - csr_matrix(D):
        to construct a CSRNDArray with a dense 2D array ``D``
            -  **D** (*array_like*) - An object exposing the array interface, an object whose             `__array__` method returns an array, or any (nested) sequence.
            - **ctx** (*Context, optional*) - Device context             (default is the current default context).
            - **dtype** (*str or numpy.dtype, optional*) - The data type of the output array.             The default dtype is ``D.dtype`` if ``D`` is an NDArray or numpy.ndarray,             float32 otherwise.

    - csr_matrix(S)
        to construct a CSRNDArray with a sparse 2D array ``S``
            -  **S** (*CSRNDArray or scipy.sparse.csr.csr_matrix*) - A sparse matrix.
            - **ctx** (*Context, optional*) - Device context             (default is the current default context).
            - **dtype** (*str or numpy.dtype, optional*) - The data type of the output array.             The default dtype is ``S.dtype``.

    - csr_matrix((M, N))
        to construct an empty CSRNDArray with shape ``(M, N)``
            -  **M** (*int*) - Number of rows in the matrix
            -  **N** (*int*) - Number of columns in the matrix
            - **ctx** (*Context, optional*) - Device context             (default is the current default context).
            - **dtype** (*str or numpy.dtype, optional*) - The data type of the output array.             The default dtype is float32.

    - csr_matrix((data, indices, indptr))
        to construct a CSRNDArray based on the definition of compressed sparse row format         using three separate arrays,         where the column indices for row i are stored in ``indices[indptr[i]:indptr[i+1]]``         and their corresponding values are stored in ``data[indptr[i]:indptr[i+1]]``.         The column indices for a given row are expected to be **sorted in ascending order.**         Duplicate column entries for the same row are not allowed.
            - **data** (*array_like*) - An object exposing the array interface, which             holds all the non-zero entries of the matrix in row-major order.
            - **indices** (*array_like*) - An object exposing the array interface, which             stores the column index for each non-zero element in ``data``.
            - **indptr** (*array_like*) - An object exposing the array interface, which             stores the offset into ``data`` of the first non-zero element number of each             row of the matrix.
            - **shape** (*tuple of int, optional*) - The shape of the array. The default             shape is inferred from the indices and indptr arrays.
            - **ctx** (*Context, optional*) - Device context             (default is the current default context).
            - **dtype** (*str or numpy.dtype, optional*) - The data type of the output array.             The default dtype is ``data.dtype`` if ``data`` is an NDArray or numpy.ndarray,             float32 otherwise.

    - csr_matrix((data, (row, col)))
        to construct a CSRNDArray based on the COOrdinate format         using three seperate arrays,         where ``row[i]`` is the row index of the element,         ``col[i]`` is the column index of the element         and ``data[i]`` is the data corresponding to the element. All the missing         elements in the input are taken to be zeroes.
            - **data** (*array_like*) - An object exposing the array interface, which             holds all the non-zero entries of the matrix in COO format.
            - **row** (*array_like*) - An object exposing the array interface, which             stores the row index for each non zero element in ``data``.
            - **col** (*array_like*) - An object exposing the array interface, which             stores the col index for each non zero element in ``data``.
            - **shape** (*tuple of int, optional*) - The shape of the array. The default             shape is inferred from the ``row`` and ``col`` arrays.
            - **ctx** (*Context, optional*) - Device context             (default is the current default context).
            - **dtype** (*str or numpy.dtype, optional*) - The data type of the output array.             The default dtype is float32.

    Parameters
    ----------
    arg1: tuple of int, tuple of array_like, array_like, CSRNDArray, scipy.sparse.csr_matrix,     scipy.sparse.coo_matrix, tuple of int or tuple of array_like
        The argument to help instantiate the csr matrix. See above for further details.
    shape : tuple of int, optional
        The shape of the csr matrix.
    ctx: Context, optional
        Device context (default is the current default context).
    dtype: str or numpy.dtype, optional
        The data type of the output array.

    Returns
    -------
    CSRNDArray
        A `CSRNDArray` with the `csr` storage representation.

    Example
    -------
    >>> a = mx.nd.sparse.csr_matrix(([1, 2, 3], [1, 0, 2], [0, 1, 2, 2, 3]), shape=(4, 3))
    >>> a.asnumpy()
    array([[ 0.,  1.,  0.],
           [ 2.,  0.,  0.],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  3.]], dtype=float32)

    See Also
    --------
    CSRNDArray : MXNet NDArray in compressed sparse row format.
    """
    if isinstance(arg1, tuple):
        arg_len = len(arg1)
        if arg_len == 2:
            if isinstance(arg1[1], tuple) and len(arg1[1]) == 2:
                data, (row, col) = arg1
                if isinstance(data, NDArray):
                    data = data.asnumpy()
                if isinstance(row, NDArray):
                    row = row.asnumpy()
                if isinstance(col, NDArray):
                    col = col.asnumpy()
                if not spsp:
                    raise ImportError('scipy could not be imported. Please make sure that the scipy is installed.')
                coo = spsp.coo_matrix((data, (row, col)), shape=shape)
                _check_shape(coo.shape, shape)
                csr = coo.tocsr()
                return array(csr, ctx=ctx, dtype=dtype)
            else:
                _check_shape(arg1, shape)
                return empty('csr', arg1, ctx=ctx, dtype=dtype)
        elif arg_len == 3:
            return _csr_matrix_from_definition(arg1[0], arg1[1], arg1[2], shape=shape, ctx=ctx, dtype=dtype)
        else:
            raise ValueError('Unexpected length of input tuple: ' + str(arg_len))
    elif isinstance(arg1, CSRNDArray) or (spsp and isinstance(arg1, spsp.csr.csr_matrix)):
        _check_shape(arg1.shape, shape)
        return array(arg1, ctx=ctx, dtype=dtype)
    elif isinstance(arg1, RowSparseNDArray):
        raise ValueError('Unexpected input type: RowSparseNDArray')
    else:
        dtype = _prepare_default_dtype(arg1, dtype)
        dns = _array(arg1, dtype=dtype)
        if ctx is not None and dns.context != ctx:
            dns = dns.as_in_context(ctx)
        _check_shape(dns.shape, shape)
        return dns.tostype('csr')