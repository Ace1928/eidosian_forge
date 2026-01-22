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
def infer_storage_type_backward(self, ograd_stype, in_stype, out_stype, igrad_stype, aux_stype):
    """infer_storage_type_backward interface. Used to infer storage
        type of inputs and outputs in the backward pass.

        Will raise an error if undefined storage type is returned.
        Returned lists have to be the same size as the input lists to infer_storage_type_backward,
        otherwise an exception will be thrown. When this interface is not implemented,
        all stypes will be inferred as default.

        Parameters
        ----------
        ograd_stype : list
            list of output gradient storage types
        in_stype : list
            list of input storage types
        out_stype : list
            list of output storage types
        igrad_stype : list
            list of input gradient storage types
        aux_stype : list
            list of auxiliary storage types

        Returns
        -------
        ograd_stype : list
            list of inferred output gradient storage types
        in_stype : list
            list of inferred input storage types
        out_stype : list
            list of inferred output storage types
        igrad_stype : list
            list of inferred input gradient storage types
        aux_stype : list
            list of inferred storage types for auxiliary states
        """
    for i, stype in enumerate(ograd_stype):
        assert stype == _STORAGE_TYPE_ID_TO_STR[_STORAGE_TYPE_DEFAULT], "Default infer_storage_type_backward implementation doesnt allow non default stypes: found non default stype '%s' for ograd_stype[%d]. Please implement infer_storage_type and infer_storage_type_backward interface in your custom operator if you have non-default output gradient stypes" % (stype, i)
    for i, stype in enumerate(igrad_stype):
        if stype == _STORAGE_TYPE_ID_TO_STR[_STORAGE_TYPE_UNDEFINED]:
            stype = _STORAGE_TYPE_ID_TO_STR[_STORAGE_TYPE_DEFAULT]
        assert stype == _STORAGE_TYPE_ID_TO_STR[_STORAGE_TYPE_DEFAULT], "Default infer_storage_type_backward implementation doesnt allow non default stypes: found non default stype '%s' for igrad_stype[%d]. Please implement infer_storage_type and infer_storage_type_backward interface in your custom operator if you have non-default input gradient stypes" % (stype, i)
    stype_lists = [ograd_stype, in_stype, out_stype, igrad_stype, aux_stype]
    for stype_list in stype_lists:
        stype_list[:] = len(stype_list) * [_STORAGE_TYPE_ID_TO_STR[_STORAGE_TYPE_DEFAULT]]
    return (stype_lists[0], stype_lists[1], stype_lists[2], stype_lists[3], stype_lists[4])