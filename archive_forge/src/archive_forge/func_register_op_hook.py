import threading
import copy
import warnings
import re
import json
from collections import OrderedDict, defaultdict
import numpy as np
from ..base import mx_real_t, MXNetError
from .. import symbol, ndarray, initializer, np_symbol
from ..symbol import Symbol, load_json
from ..ndarray import NDArray
from .. import name as _name
from .parameter import Parameter, ParameterDict, DeferredInitializationError
from .utils import _indent, _brief_print_list, HookHandle
from .utils import _check_same_symbol_type, _check_all_np_ndarrays
from .. import numpy_extension as _mx_npx
from .. import numpy as _mx_np
from .. util import is_np_array, np_shape, np_array
def register_op_hook(self, callback, monitor_all=False):
    """Install op hook for block recursively.

        Parameters
        ----------
        callback : function
            Takes a string and a NDArrayHandle.
        monitor_all : bool, default False
            If true, monitor both input and output, otherwise monitor output only.
        """
    self._callback = callback
    self._monitor_all = monitor_all
    for cld in self._children.values():
        cld._callback = callback
        cld._monitor_all = monitor_all