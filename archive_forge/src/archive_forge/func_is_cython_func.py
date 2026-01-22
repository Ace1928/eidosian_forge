import functools
import importlib
import importlib.util
import inspect
import itertools
import logging
import os
import pkgutil
import re
import shlex
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import warnings
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import (
import catalogue
import langcodes
import numpy
import srsly
import thinc
from catalogue import Registry, RegistryError
from packaging.requirements import Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version
from thinc.api import (
from thinc.api import compounding, decaying, fix_random_seed  # noqa: F401
from . import about
from .compat import CudaStream, cupy, importlib_metadata, is_windows
from .errors import OLD_MODEL_SHORTCUTS, Errors, Warnings
from .symbols import ORTH
def is_cython_func(func: Callable) -> bool:
    """Slightly hacky check for whether a callable is implemented in Cython.
    Can be used to implement slightly different behaviors, especially around
    inspecting and parameter annotations. Note that this will only return True
    for actual cdef functions and methods, not regular Python functions defined
    in Python modules.

    func (Callable): The callable to check.
    RETURNS (bool): Whether the callable is Cython (probably).
    """
    attr = '__pyx_vtable__'
    if hasattr(func, attr):
        return True
    if hasattr(func, '__qualname__') and hasattr(func, '__module__') and (func.__module__ in sys.modules):
        cls_func = vars(sys.modules[func.__module__])[func.__qualname__.split('.')[0]]
        return hasattr(cls_func, attr)
    return False