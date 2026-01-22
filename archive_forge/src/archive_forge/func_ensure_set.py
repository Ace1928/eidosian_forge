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
def ensure_set(s: Set[T], *, copy: bool=False) -> set[T]:
    """Convert a generic Set into a set.

    Parameters
    ----------
    s : Set
    copy : bool
        If True, guarantee that the return value is always a shallow copy of s;
        otherwise it may be the input itself.
    """
    if type(s) is set:
        return s.copy() if copy else s
    return set(s)