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
@classmethod
def get_suitable_cache_subpath(cls, py_file):
    """Given the Python file path, compute a suitable path inside the
        cache directory.

        This will reduce a file path that is too long, which can be a problem
        on some operating system (i.e. Windows 7).
        """
    path = os.path.abspath(py_file)
    subpath = os.path.dirname(path)
    parentdir = os.path.split(subpath)[-1]
    hashed = hashlib.sha1(subpath.encode()).hexdigest()
    return '_'.join([parentdir, hashed])