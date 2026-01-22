import fnmatch
import functools
import inspect
import os
import warnings
from io import IOBase
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from torch.utils.data._utils.serialization import DILL_AVAILABLE
@classmethod
def close_streams(cls, v, depth=0):
    """Traverse structure and attempts to close all found StreamWrappers on best effort basis."""
    if depth > 10:
        return
    if isinstance(v, StreamWrapper):
        v.close()
    elif isinstance(v, dict):
        for vv in v.values():
            cls.close_streams(vv, depth=depth + 1)
    elif isinstance(v, (list, tuple)):
        for vv in v:
            cls.close_streams(vv, depth=depth + 1)