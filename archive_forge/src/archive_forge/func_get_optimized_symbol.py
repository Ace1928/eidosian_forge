from array import array as py_array
import ctypes
import copy
import numpy as np
from .base import _LIB
from .base import mx_uint, NDArrayHandle, SymbolHandle, ExecutorHandle, py_str, mx_int
from .base import check_call, c_handle_array, c_array_buf, c_str_array
from . import ndarray
from .ndarray import NDArray
from .ndarray import _ndarray_cls
from .executor_manager import _split_input_slice, _check_arguments, _load_data, _load_label
def get_optimized_symbol(self):
    """Get an optimized version of the symbol from the executor.

        Returns
        -------
        symbol : Symbol
            Optimized symbol from the executor.
        """
    from .symbol import Symbol
    sym_handle = SymbolHandle()
    check_call(_LIB.MXExecutorGetOptimizedSymbol(self.handle, ctypes.byref(sym_handle)))
    ret = Symbol(sym_handle)
    return ret