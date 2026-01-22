import ctypes
import json
import logging
import pickle
from enum import IntEnum, unique
from typing import Any, Dict, List
import numpy as np
from ._typing import _T
from .core import _LIB, _check_call, c_str, from_pystr_to_cstr, py_str
def communicator_print(msg: Any) -> None:
    """Print message to the communicator.

    This function can be used to communicate the information of
    the progress to the communicator.

    Parameters
    ----------
    msg : str
        The message to be printed to the communicator.
    """
    if not isinstance(msg, str):
        msg = str(msg)
    is_dist = _LIB.XGCommunicatorIsDistributed()
    if is_dist != 0:
        _check_call(_LIB.XGCommunicatorPrint(c_str(msg.strip())))
    else:
        print(msg.strip(), flush=True)