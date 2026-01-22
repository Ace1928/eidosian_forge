from array import array
import ctypes
import warnings
from numbers import Number
import numpy as _numpy  # pylint: disable=relative-import
from ..attribute import AttrScope
from ..base import _LIB, numeric_types, c_array, c_array_buf, c_str, c_str_array, c_handle_array
from ..base import mx_uint, py_str, string_types, integer_types, mx_int, mx_int64
from ..base import NDArrayHandle, ExecutorHandle, SymbolHandle
from ..base import check_call, MXNetError, NotImplementedForSymbol
from ..context import Context, current_context
from ..ndarray import NDArray, _DTYPE_NP_TO_MX, _DTYPE_MX_TO_NP, _GRAD_REQ_MAP
from ..ndarray.ndarray import _STORAGE_TYPE_STR_TO_ID, _int64_enabled, _SIGNED_INT32_UPPER_LIMIT
from ..ndarray import _ndarray_cls
from ..executor import Executor
from . import _internal
from . import op
from ._internal import SymbolBase, _set_symbol_class
from ..util import is_np_shape
def list_auxiliary_states(self):
    """Lists all the auxiliary states in the symbol.

        Example
        -------
        >>> a = mx.sym.var('a')
        >>> b = mx.sym.var('b')
        >>> c = a + b
        >>> c.list_auxiliary_states()
        []

        Example of auxiliary states in `BatchNorm`.

        >>> data = mx.symbol.Variable('data')
        >>> weight = mx.sym.Variable(name='fc1_weight')
        >>> fc1  = mx.symbol.FullyConnected(data = data, weight=weight, name='fc1', num_hidden=128)
        >>> fc2 = mx.symbol.BatchNorm(fc1, name='batchnorm0')
        >>> fc2.list_auxiliary_states()
        ['batchnorm0_moving_mean', 'batchnorm0_moving_var']

        Returns
        -------
        aux_states : list of str
            List of the auxiliary states in input symbol.

        Notes
        -----
        Auxiliary states are special states of symbols that do not correspond to an argument,
        and are not updated by gradient descent. Common examples of auxiliary states
        include the `moving_mean` and `moving_variance` in `BatchNorm`.
        Most operators do not have auxiliary states.
        """
    size = ctypes.c_uint()
    sarr = ctypes.POINTER(ctypes.c_char_p)()
    check_call(_LIB.MXSymbolListAuxiliaryStates(self.handle, ctypes.byref(size), ctypes.byref(sarr)))
    return [py_str(sarr[i]) for i in range(size.value)]