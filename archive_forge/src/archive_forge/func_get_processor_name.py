import ctypes
import json
import logging
import pickle
from enum import IntEnum, unique
from typing import Any, Dict, List
import numpy as np
from ._typing import _T
from .core import _LIB, _check_call, c_str, from_pystr_to_cstr, py_str
def get_processor_name() -> str:
    """Get the processor name.

    Returns
    -------
    name : str
        the name of processor(host)
    """
    name_str = ctypes.c_char_p()
    _check_call(_LIB.XGCommunicatorGetProcessorName(ctypes.byref(name_str)))
    value = name_str.value
    assert value
    return py_str(value)