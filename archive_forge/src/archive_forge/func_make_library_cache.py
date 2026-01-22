from abc import ABCMeta, abstractmethod, abstractproperty
import contextlib
import errno
import hashlib
import inspect
import itertools
import os
import pickle
import sys
import tempfile
import uuid
import warnings
from numba.misc.appdirs import AppDirs
import numba
from numba.core.errors import NumbaWarning
from numba.core.base import BaseContext
from numba.core.codegen import CodeLibrary
from numba.core.compiler import CompileResult
from numba.core import config, compiler
from numba.core.serialize import dumps
def make_library_cache(prefix):
    """
    Create a Cache class for additional compilation features to cache their
    result for reuse.  The cache is saved in filename pattern like
    in ``FunctionCache`` but with additional *prefix* as specified.
    """
    assert prefix not in _lib_cache_prefixes
    _lib_cache_prefixes.add(prefix)

    class CustomCodeLibraryCacheImpl(CodeLibraryCacheImpl):
        _filename_prefix = prefix

    class LibraryCache(Cache):
        """
        Implements Cache that saves and loads CodeLibrary objects for additional
        feature for the specified python function.
        """
        _impl_class = CustomCodeLibraryCacheImpl
    return LibraryCache