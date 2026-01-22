import functools
import inspect
import platform
import sys
import types
from importlib import import_module
from typing import List, TypeVar
import distutils.filelist
def get_unpatched(item: _T) -> _T:
    lookup = get_unpatched_class if isinstance(item, type) else get_unpatched_function if isinstance(item, types.FunctionType) else lambda item: None
    return lookup(item)