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
def get_source_info(cls):
    """Return a dictionary with information about the source code of the
        implementation.

        Returns
        -------
        info : dict
            - "kind" : str
                The implementation kind.
            - "name" : str
                The name of the function that provided the definition.
            - "sig" : str
                The formatted signature of the function.
            - "filename" : str
                The name of the source file.
            - "lines": tuple (int, int)
                First and list line number.
            - "docstring": str
                The docstring of the definition.
        """
    basepath = os.path.dirname(os.path.dirname(numba.__file__))
    impl = cls._overload_func
    code, firstlineno, path = cls.get_source_code_info(impl)
    sig = str(utils.pysignature(impl))
    info = {'kind': 'overload', 'name': getattr(impl, '__qualname__', impl.__name__), 'sig': sig, 'filename': utils.safe_relpath(path, start=basepath), 'lines': (firstlineno, firstlineno + len(code) - 1), 'docstring': impl.__doc__}
    return info