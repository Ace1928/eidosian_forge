from __future__ import annotations
import dataclasses
import datetime
import decimal
import hashlib
import inspect
import pathlib
import pickle
import types
import uuid
import warnings
from collections import OrderedDict
from collections.abc import Hashable, Iterable, Iterator, Mapping
from concurrent.futures import Executor
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from functools import partial
from numbers import Integral, Number
from operator import getitem
from typing import Any, Literal, TypeVar
import cloudpickle
from tlz import curry, groupby, identity, merge
from tlz.functoolz import Compose
from dask import config, local
from dask._compatibility import EMSCRIPTEN
from dask.core import flatten
from dask.core import get as simple_get
from dask.core import literal, quote
from dask.hashing import hash_buffer_hex
from dask.system import CPU_COUNT
from dask.typing import Key, SchedulerGetCallable
from dask.utils import (
@normalize_token.register(object)
def normalize_object(o):
    method = getattr(o, '__dask_tokenize__', None)
    if method is not None and (not isinstance(o, type)):
        return method()
    if type(o) is object:
        return _normalize_pure_object(o)
    if dataclasses.is_dataclass(o) and (not isinstance(o, type)):
        return _normalize_dataclass(o)
    try:
        return _normalize_pickle(o)
    except Exception:
        _maybe_raise_nondeterministic(f'Object {o!r} cannot be deterministically hashed. See https://docs.dask.org/en/latest/custom-collections.html#implementing-deterministic-hashing for more information.')
        return uuid.uuid4().hex