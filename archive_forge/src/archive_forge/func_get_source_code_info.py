from abc import ABC, abstractmethod
import functools
import sys
import inspect
import os.path
from collections import namedtuple
from collections.abc import Sequence
from types import MethodType, FunctionType, MappingProxyType
import numba
from numba.core import types, utils, targetconfig
from numba.core.errors import (
from numba.core.cpu_options import InlineOptions
@classmethod
def get_source_code_info(cls, impl):
    """
        Gets the source information about function impl.
        Returns:

        code - str: source code as a string
        firstlineno - int: the first line number of the function impl
        path - str: the path to file containing impl

        if any of the above are not available something generic is returned
        """
    try:
        code, firstlineno = inspect.getsourcelines(impl)
    except OSError:
        code = 'None available (built from string?)'
        firstlineno = 0
    path = inspect.getsourcefile(impl)
    if path is None:
        path = '<unknown> (built from string?)'
    return (code, firstlineno, path)