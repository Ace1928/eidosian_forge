from __future__ import annotations
import os
import warnings
import numpy as np
from numpy.testing import assert_
import scipy._lib.array_api_compat.array_api_compat as array_api_compat
from scipy._lib.array_api_compat.array_api_compat import size
import scipy._lib.array_api_compat.array_api_compat.numpy as array_api_compat_numpy
def compliance_scipy(arrays):
    """Raise exceptions on known-bad subclasses.

    The following subclasses are not supported and raise and error:
    - `numpy.ma.MaskedArray`
    - `numpy.matrix`
    - NumPy arrays which do not have a boolean or numerical dtype
    - Any array-like which is neither array API compatible nor coercible by NumPy
    - Any array-like which is coerced by NumPy to an unsupported dtype
    """
    for i in range(len(arrays)):
        array = arrays[i]
        if isinstance(array, np.ma.MaskedArray):
            raise TypeError('Inputs of type `numpy.ma.MaskedArray` are not supported.')
        elif isinstance(array, np.matrix):
            raise TypeError('Inputs of type `numpy.matrix` are not supported.')
        if isinstance(array, (np.ndarray, np.generic)):
            dtype = array.dtype
            if not (np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)):
                raise TypeError(f'An argument has dtype `{dtype!r}`; only boolean and numerical dtypes are supported.')
        elif not array_api_compat.is_array_api_obj(array):
            try:
                array = np.asanyarray(array)
            except TypeError:
                raise TypeError('An argument is neither array API compatible nor coercible by NumPy.')
            dtype = array.dtype
            if not (np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)):
                message = f'An argument was coerced to an unsupported dtype `{dtype!r}`; only boolean and numerical dtypes are supported.'
                raise TypeError(message)
            arrays[i] = array
    return arrays