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
@contextmanager
def filetexts(d, open=open, mode='t', use_tmpdir=True):
    """Dumps a number of textfiles to disk

    Parameters
    ----------
    d : dict
        a mapping from filename to text like {'a.csv': '1,1
2,2'}

    Since this is meant for use in tests, this context manager will
    automatically switch to a temporary current directory, to avoid
    race conditions when running tests in parallel.
    """
    with tmp_cwd() if use_tmpdir else nullcontext():
        for filename, text in d.items():
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError:
                pass
            f = open(filename, 'w' + mode)
            try:
                f.write(text)
            finally:
                try:
                    f.close()
                except AttributeError:
                    pass
        yield list(d)
        for filename in d:
            if os.path.exists(filename):
                with suppress(OSError):
                    os.remove(filename)