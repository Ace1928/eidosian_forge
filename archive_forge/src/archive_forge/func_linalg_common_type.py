import os
import numpy
from numpy import linalg
import cupy
import cupy._util
from cupy import _core
import cupyx
def linalg_common_type(*arrays, reject_float16=True):
    """Common type for linalg

    The logic is intended to be equivalent with
    `numpy.linalg.linalg._commonType`.
    The differences from `numpy.common_type` are
    - to accept ``bool_`` arrays, and
    - to reject ``float16`` arrays.

    Args:
        *arrays (ndarray): Input arrays.
        reject_float16 (bool): Flag to follow NumPy to raise TypeError for
            ``float16`` inputs.

    Returns:
        compute_dtype (dtype): The precision to be used in linalg calls.
        result_dtype (dtype): The dtype of (possibly complex) output(s).
    """
    dtypes = [arr.dtype for arr in arrays]
    if reject_float16 and 'float16' in dtypes:
        raise TypeError('float16 is unsupported in linalg')
    if _default_precision is not None:
        cupy._util.experimental('CUPY_DEFAULT_PRECISION')
        if _default_precision not in ('32', '64'):
            raise ValueError('invalid CUPY_DEFAULT_PRECISION: {}'.format(_default_precision))
        default = 'float' + _default_precision
    else:
        default = 'float64'
    compute_dtype = _common_type_internal(default, *dtypes)
    if compute_dtype == 'float16':
        compute_dtype = numpy.dtype('float32')
    result_dtype = _common_type_internal('float64', *dtypes)
    return (compute_dtype, result_dtype)