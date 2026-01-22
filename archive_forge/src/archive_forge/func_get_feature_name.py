import abc
import ctypes
import inspect
import json
import warnings
from collections import OrderedDict
from copy import deepcopy
from enum import Enum
from functools import wraps
from os import SEEK_END, environ
from os.path import getsize
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
import scipy.sparse
from .compat import (PANDAS_INSTALLED, PYARROW_INSTALLED, arrow_cffi, arrow_is_floating, arrow_is_integer, concat,
from .libpath import find_lib_path
def get_feature_name(self) -> List[str]:
    """Get the names of columns (features) in the Dataset.

        Returns
        -------
        feature_names : list of str
            The names of columns (features) in the Dataset.
        """
    if self._handle is None:
        raise LightGBMError('Cannot get feature_name before construct dataset')
    num_feature = self.num_feature()
    tmp_out_len = ctypes.c_int(0)
    reserved_string_buffer_size = 255
    required_string_buffer_size = ctypes.c_size_t(0)
    string_buffers = [ctypes.create_string_buffer(reserved_string_buffer_size) for _ in range(num_feature)]
    ptr_string_buffers = (ctypes.c_char_p * num_feature)(*map(ctypes.addressof, string_buffers))
    _safe_call(_LIB.LGBM_DatasetGetFeatureNames(self._handle, ctypes.c_int(num_feature), ctypes.byref(tmp_out_len), ctypes.c_size_t(reserved_string_buffer_size), ctypes.byref(required_string_buffer_size), ptr_string_buffers))
    if num_feature != tmp_out_len.value:
        raise ValueError("Length of feature names doesn't equal with num_feature")
    actual_string_buffer_size = required_string_buffer_size.value
    if reserved_string_buffer_size < actual_string_buffer_size:
        string_buffers = [ctypes.create_string_buffer(actual_string_buffer_size) for _ in range(num_feature)]
        ptr_string_buffers = (ctypes.c_char_p * num_feature)(*map(ctypes.addressof, string_buffers))
        _safe_call(_LIB.LGBM_DatasetGetFeatureNames(self._handle, ctypes.c_int(num_feature), ctypes.byref(tmp_out_len), ctypes.c_size_t(actual_string_buffer_size), ctypes.byref(required_string_buffer_size), ptr_string_buffers))
    return [string_buffers[i].value.decode('utf-8') for i in range(num_feature)]