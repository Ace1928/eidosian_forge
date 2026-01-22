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
def is_same_func(func1: Callable, func2: Callable) -> bool:
    """Approximately decide whether two functions are the same, even if their
    identity is different (e.g. after they have been live reloaded). Mostly
    used in the @Language.component and @Language.factory decorators to decide
    whether to raise if a factory already exists. Allows decorator to run
    multiple times with the same function.

    func1 (Callable): The first function.
    func2 (Callable): The second function.
    RETURNS (bool): Whether it's the same function (most likely).
    """
    if not callable(func1) or not callable(func2):
        return False
    if not hasattr(func1, '__qualname__') or not hasattr(func2, '__qualname__'):
        return False
    same_name = func1.__qualname__ == func2.__qualname__
    same_file = inspect.getfile(func1) == inspect.getfile(func2)
    same_code = inspect.getsourcelines(func1) == inspect.getsourcelines(func2)
    return same_name and same_file and same_code