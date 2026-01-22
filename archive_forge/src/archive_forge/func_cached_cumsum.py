from __future__ import annotations
import codecs
import functools
import inspect
import os
import re
import shutil
import sys
import tempfile
import types
import uuid
import warnings
from collections.abc import Hashable, Iterable, Iterator, Mapping, Set
from contextlib import contextmanager, nullcontext, suppress
from datetime import datetime, timedelta
from errno import ENOENT
from functools import lru_cache, wraps
from importlib import import_module
from numbers import Integral, Number
from operator import add
from threading import Lock
from typing import Any, Callable, ClassVar, Literal, TypeVar, cast, overload
from weakref import WeakValueDictionary
import tlz as toolz
from dask import config
from dask.core import get_deps
from dask.typing import no_default
def cached_cumsum(seq, initial_zero=False):
    """Compute :meth:`toolz.accumulate` with caching.

    Caching is by the identify of `seq` rather than the value. It is thus
    important that `seq` is a tuple of immutable objects, and this function
    is intended for use where `seq` is a value that will persist (generally
    block sizes).

    Parameters
    ----------
    seq : tuple
        Values to cumulatively sum.
    initial_zero : bool, optional
        If true, the return value is prefixed with a zero.

    Returns
    -------
    tuple
    """
    if isinstance(seq, tuple):
        result = _cumsum(_HashIdWrapper(seq), initial_zero)
    else:
        result = _cumsum(tuple(seq), initial_zero)
    return result