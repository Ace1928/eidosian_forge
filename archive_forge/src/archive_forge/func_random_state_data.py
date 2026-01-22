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
def random_state_data(n: int, random_state=None) -> list:
    """Return a list of arrays that can initialize
    ``np.random.RandomState``.

    Parameters
    ----------
    n : int
        Number of arrays to return.
    random_state : int or np.random.RandomState, optional
        If an int, is used to seed a new ``RandomState``.
    """
    import numpy as np
    if not all((hasattr(random_state, attr) for attr in ['normal', 'beta', 'bytes', 'uniform'])):
        random_state = np.random.RandomState(random_state)
    random_data = random_state.bytes(624 * n * 4)
    l = list(np.frombuffer(random_data, dtype='<u4').reshape((n, -1)))
    assert len(l) == n
    return l