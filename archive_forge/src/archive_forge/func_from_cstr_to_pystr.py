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
def from_cstr_to_pystr(data: CStrPptr, length: c_bst_ulong) -> List[str]:
    """Revert C pointer to Python str

    Parameters
    ----------
    data :
        pointer to data
    length :
        pointer to length of data
    """
    res = []
    for i in range(length.value):
        try:
            res.append(str(cast(bytes, data[i]).decode('ascii')))
        except UnicodeDecodeError:
            res.append(str(cast(bytes, data[i]).decode('utf-8')))
    return res