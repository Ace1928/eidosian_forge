import functools
import time
import inspect
import collections
import types
import itertools
import warnings
import setuptools.extern.more_itertools
from typing import Callable, TypeVar
def call_aside(*args, **kwargs):
    """
    Deprecated name for invoke.
    """
    warnings.warn('call_aside is deprecated, use invoke', DeprecationWarning)
    return invoke(*args, **kwargs)