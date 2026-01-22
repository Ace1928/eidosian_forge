import fnmatch
import functools
import inspect
import os
import warnings
from io import IOBase
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from torch.utils.data._utils.serialization import DILL_AVAILABLE
def validate_pathname_binary_tuple(data: Tuple[str, IOBase]):
    if not isinstance(data, tuple):
        raise TypeError(f'pathname binary data should be tuple type, but it is type {type(data)}')
    if len(data) != 2:
        raise TypeError(f'pathname binary stream tuple length should be 2, but got {len(data)}')
    if not isinstance(data[0], str):
        raise TypeError(f'pathname within the tuple should have string type pathname, but it is type {type(data[0])}')
    if not isinstance(data[1], IOBase) and (not isinstance(data[1], StreamWrapper)):
        raise TypeError(f'binary stream within the tuple should have IOBase orits subclasses as type, but it is type {type(data[1])}')