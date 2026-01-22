from array import array as native_array
import ctypes
import warnings
import operator
from functools import reduce # pylint: disable=redefined-builtin
import numpy as np
from ..base import _LIB, numeric_types, integer_types
from ..base import c_str, c_array, c_array_buf, c_handle_array, mx_real_t
from ..base import mx_uint, NDArrayHandle, check_call, DLPackHandle, mx_int, mx_int64
from ..base import ctypes2buffer
from ..runtime import Features
from ..context import Context, current_context
from ..util import is_np_array
from . import _internal
from . import op
from ._internal import NDArrayBase
def to_dlpack_for_read(self):
    """Returns a reference view of NDArray that represents as DLManagedTensor until
        all previous write operations on the current array are finished.

        Returns
        -------
        PyCapsule (the pointer of DLManagedTensor)
            a reference view of NDArray that represents as DLManagedTensor.

        Examples
        --------
        >>> x = mx.nd.ones((2,3))
        >>> y = mx.nd.to_dlpack_for_read(x)
        >>> type(y)
        <class 'PyCapsule'>
        >>> z = mx.nd.from_dlpack(y)
        >>> z
        [[1. 1. 1.]
         [1. 1. 1.]]
        <NDArray 2x3 @cpu(0)>
        """
    return to_dlpack_for_read(self)