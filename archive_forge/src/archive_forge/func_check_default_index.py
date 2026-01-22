import contextlib
from functools import wraps
import os
import os.path as osp
import struct
import tempfile
from types import TracebackType
from typing import Any, Callable, TYPE_CHECKING, Optional, Type
from git.types import Literal, PathLike, _T
@wraps(func)
def check_default_index(self: 'IndexFile', *args: Any, **kwargs: Any) -> _T:
    if self._file_path != self._index_path():
        raise AssertionError('Cannot call %r on indices that do not represent the default git index' % func.__name__)
    return func(self, *args, **kwargs)