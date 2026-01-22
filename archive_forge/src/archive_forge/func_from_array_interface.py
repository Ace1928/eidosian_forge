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
def from_array_interface(interface: dict) -> NumpyOrCupy:
    """Convert array interface to numpy or cupy array"""

    class Array:
        """Wrapper type for communicating with numpy and cupy."""
        _interface: Optional[dict] = None

        @property
        def __array_interface__(self) -> Optional[dict]:
            return self._interface

        @__array_interface__.setter
        def __array_interface__(self, interface: dict) -> None:
            self._interface = copy.copy(interface)
            self._interface['shape'] = tuple(self._interface['shape'])
            self._interface['data'] = tuple(self._interface['data'])
            if self._interface.get('strides', None) is not None:
                self._interface['strides'] = tuple(self._interface['strides'])

        @property
        def __cuda_array_interface__(self) -> Optional[dict]:
            return self.__array_interface__

        @__cuda_array_interface__.setter
        def __cuda_array_interface__(self, interface: dict) -> None:
            self.__array_interface__ = interface
    arr = Array()
    if 'stream' in interface:
        spec = importlib.util.find_spec('cupy')
        if spec is None:
            raise ImportError('`cupy` is required for handling CUDA buffer.')
        import cupy as cp
        arr.__cuda_array_interface__ = interface
        out = cp.array(arr, copy=True)
    else:
        arr.__array_interface__ = interface
        out = np.array(arr, copy=True)
    return out