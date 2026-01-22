from __future__ import annotations
import abc
import contextlib, os.path, re
import enum
import itertools
import typing as T
from functools import lru_cache
from .. import coredata
from .. import mlog
from .. import mesonlib
from ..mesonlib import (
from ..arglist import CompilerArgs
@lru_cache(maxsize=None)
def cached_by_name(fname: 'mesonlib.FileOrString') -> bool:
    suffix = fname.split('.')[-1]
    return suffix in obj_suffixes