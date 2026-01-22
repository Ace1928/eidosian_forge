from array import array
from threading import Lock
import traceback
import ctypes
from ctypes import c_int, c_void_p, CFUNCTYPE, POINTER, cast
from .base import _LIB, check_call, string_types, mx_uint
from .base import NDArrayHandle, c_array, c_handle_array, c_array_buf, MXCallbackList, SymbolHandle
from .ndarray import NDArray, _ndarray_cls
from .ndarray import _GRAD_REQ_MAP
from .symbol import Symbol
def backward_entry(num_ograds, num_igrads, ptrs, reqs, is_train, _):
    """entry point for backward."""
    try:
        output_grads = [NDArray(ctypes.cast(i, NDArrayHandle), writable=False) for i in ptrs[:num_ograds]]
        input_grads = [NDArray(ctypes.cast(i, NDArrayHandle), writable=True) for i in ptrs[num_ograds:num_ograds + num_igrads]]
        reqs = [reqs[i] for i in range(num_igrads)]
        rets = self.backward(*output_grads)
        if isinstance(rets, NDArray):
            rets = (rets,)
        assert len(rets) == len(input_grads), '%s.backward must return exactly the same number of NDArrays as the number of NDArrays arguments to forward.Expecting %d got %d' % (self.__class__.name, len(input_grads), len(rets))
        for igrad, ret, req in zip(input_grads, rets, reqs):
            assert isinstance(ret, NDArray), 'autograd.Function.backward must return NDArrays, not %s' % type(ret)
            if req == 0:
                return True
            elif req in (1, 2):
                igrad[:] = ret
            elif req == 'add':
                igrad[:] += ret
    except Exception:
        print('Error in Function.backward: %s' % traceback.format_exc())
        return False
    return True