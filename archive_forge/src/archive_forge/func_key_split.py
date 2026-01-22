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
@functools.lru_cache(100000)
def key_split(s):
    """
    >>> key_split('x')
    'x'
    >>> key_split('x-1')
    'x'
    >>> key_split('x-1-2-3')
    'x'
    >>> key_split(('x-2', 1))
    'x'
    >>> key_split("('x-2', 1)")
    'x'
    >>> key_split("('x', 1)")
    'x'
    >>> key_split('hello-world-1')
    'hello-world'
    >>> key_split(b'hello-world-1')
    'hello-world'
    >>> key_split('ae05086432ca935f6eba409a8ecd4896')
    'data'
    >>> key_split('<module.submodule.myclass object at 0xdaf372')
    'myclass'
    >>> key_split(None)
    'Other'
    >>> key_split('x-abcdefab')  # ignores hex
    'x'
    >>> key_split('_(x)')  # strips unpleasant characters
    'x'
    """
    if type(s) is bytes:
        return key_split(s.decode())
    if type(s) is tuple:
        return key_split(s[0])
    try:
        words = s.split('-')
        if not words[0][0].isalpha():
            result = words[0].split(',')[0].strip('_\'()"')
        else:
            result = words[0]
        for word in words[1:]:
            if word.isalpha() and (not (len(word) == 8 and hex_pattern.match(word) is not None)):
                result += '-' + word
            else:
                break
        if len(result) == 32 and re.match('[a-f0-9]{32}', result):
            return 'data'
        else:
            if result[0] == '<':
                result = result.strip('<>').split()[0].split('.')[-1]
            return sys.intern(result)
    except Exception:
        return 'Other'