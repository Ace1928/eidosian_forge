import traceback
import warnings
import collections
from array import array
from threading import Lock
import ctypes
from ctypes import CFUNCTYPE, POINTER, Structure, pointer
from ctypes import c_void_p, c_int, c_char, c_char_p, cast, c_bool
from .base import _LIB, check_call, MXCallbackList, c_array, c_array_buf, mx_int, OpHandle
from .base import c_str, mx_uint, mx_float, ctypes2numpy_shared, NDArrayHandle, py_str
from . import symbol, context
from .ndarray import NDArray, _DTYPE_NP_TO_MX, _DTYPE_MX_TO_NP
from .ndarray.ndarray import _STORAGE_TYPE_STR_TO_ID, _STORAGE_TYPE_ID_TO_STR
from .ndarray.ndarray import _STORAGE_TYPE_UNDEFINED, _STORAGE_TYPE_DEFAULT
from .ndarray.ndarray import _STORAGE_TYPE_CSR, _STORAGE_TYPE_ROW_SPARSE
from .ndarray import _ndarray_cls
from .numpy.multiarray import _np_ndarray_cls
from .util import is_np_array
def list_arguments_entry(out, _):
    """C Callback for CustomOpProp::ListArguments"""
    try:
        ret = op_prop.list_arguments()
        ret = [c_str(i) for i in ret] + [c_char_p(0)]
        ret = c_array(c_char_p, ret)
        out[0] = cast(ret, POINTER(POINTER(c_char)))
        list_arguments_entry._ref_holder = [out]
    except Exception:
        print('Error in %s.list_arguments: %s' % (reg_name, traceback.format_exc()))
        return False
    return True