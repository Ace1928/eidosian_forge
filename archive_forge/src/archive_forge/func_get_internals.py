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
def get_internals(self):
    """Gets a new grouped symbol `sgroup`. The output of `sgroup` is a list of
        outputs of all of the internal nodes.

        Consider the following code:

        Example
        -------
        >>> a = mx.sym.var('a')
        >>> b = mx.sym.var('b')
        >>> c = a + b
        >>> d = c.get_internals()
        >>> d
        <Symbol Grouped>
        >>> d.list_outputs()
        ['a', 'b', '_plus4_output']

        Returns
        -------
        sgroup : Symbol
            A symbol group containing all internal and leaf nodes of the computation graph
            used to compute the symbol.
        """
    handle = SymbolHandle()
    check_call(_LIB.MXSymbolGetInternals(self.handle, ctypes.byref(handle)))
    return Symbol(handle=handle)