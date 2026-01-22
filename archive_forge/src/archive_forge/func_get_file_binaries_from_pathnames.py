import fnmatch
import functools
import inspect
import os
import warnings
from io import IOBase
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from torch.utils.data._utils.serialization import DILL_AVAILABLE
def get_file_binaries_from_pathnames(pathnames: Iterable, mode: str, encoding: Optional[str]=None):
    if not isinstance(pathnames, Iterable):
        pathnames = [pathnames]
    if mode in ('b', 't'):
        mode = 'r' + mode
    for pathname in pathnames:
        if not isinstance(pathname, str):
            raise TypeError(f'Expected string type for pathname, but got {type(pathname)}')
        yield (pathname, StreamWrapper(open(pathname, mode, encoding=encoding)))