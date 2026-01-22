import functools
import inspect
import platform
import sys
import types
from importlib import import_module
from typing import List, TypeVar
import distutils.filelist
def get_unpatched_function(candidate):
    return candidate.unpatched