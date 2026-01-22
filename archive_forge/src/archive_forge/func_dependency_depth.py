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
def dependency_depth(dsk):
    deps, _ = get_deps(dsk)

    @lru_cache(maxsize=None)
    def max_depth_by_deps(key):
        if not deps[key]:
            return 1
        d = 1 + max((max_depth_by_deps(dep_key) for dep_key in deps[key]))
        return d
    return max((max_depth_by_deps(dep_key) for dep_key in deps.keys()))