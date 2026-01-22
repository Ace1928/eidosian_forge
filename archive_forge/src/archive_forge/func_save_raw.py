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
def save_raw(self, raw_format: str='deprecated') -> bytearray:
    """Save the model to a in memory buffer representation instead of file.

        Parameters
        ----------
        raw_format :
            Format of output buffer. Can be `json`, `ubj` or `deprecated`.  Right now
            the default is `deprecated` but it will be changed to `ubj` (univeral binary
            json) in the future.

        Returns
        -------
        An in memory buffer representation of the model
        """
    length = c_bst_ulong()
    cptr = ctypes.POINTER(ctypes.c_char)()
    config = make_jcargs(format=raw_format)
    _check_call(_LIB.XGBoosterSaveModelToBuffer(self.handle, config, ctypes.byref(length), ctypes.byref(cptr)))
    return ctypes2buffer(cptr, length.value)