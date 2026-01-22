import copy
import ctypes
import importlib.util
import json
import os
import re
import sys
import warnings
import weakref
from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import IntEnum, unique
from functools import wraps
from inspect import Parameter, signature
from typing import (
import numpy as np
import scipy.sparse
from ._typing import (
from .compat import PANDAS_INSTALLED, DataFrame, py_str
from .libpath import find_lib_path
def get_quantile_cut(self) -> Tuple[np.ndarray, np.ndarray]:
    """Get quantile cuts for quantization.

        .. versionadded:: 2.0.0

        """
    n_features = self.num_col()
    c_sindptr = ctypes.c_char_p()
    c_sdata = ctypes.c_char_p()
    config = make_jcargs()
    _check_call(_LIB.XGDMatrixGetQuantileCut(self.handle, config, ctypes.byref(c_sindptr), ctypes.byref(c_sdata)))
    assert c_sindptr.value is not None
    assert c_sdata.value is not None
    i_indptr = json.loads(c_sindptr.value)
    indptr = from_array_interface(i_indptr)
    assert indptr.size == n_features + 1
    assert indptr.dtype == np.uint64
    i_data = json.loads(c_sdata.value)
    data = from_array_interface(i_data)
    assert data.size == indptr[-1]
    assert data.dtype == np.float32
    return (indptr, data)