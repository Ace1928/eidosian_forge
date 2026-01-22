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
def clone_key(key: KeyOrStrT, seed: Hashable) -> KeyOrStrT:
    """Clone a key from a Dask collection, producing a new key with the same prefix and
    indices and a token which is a deterministic function of the previous key and seed.

    Examples
    --------
    >>> clone_key("x", 123)
    'x-c4fb64ccca807af85082413d7ef01721'
    >>> clone_key("inc-cbb1eca3bafafbb3e8b2419c4eebb387", 123)
    'inc-bc629c23014a4472e18b575fdaf29ee7'
    >>> clone_key(("sum-cbb1eca3bafafbb3e8b2419c4eebb387", 4, 3), 123)
    ('sum-c053f3774e09bd0f7de6044dbc40e71d', 4, 3)
    """
    if isinstance(key, tuple) and key and isinstance(key[0], str):
        return (clone_key(key[0], seed),) + key[1:]
    if isinstance(key, str):
        prefix = key_split(key)
        return prefix + '-' + tokenize(key, seed)
    raise TypeError(f'Expected str or a tuple starting with str; got {key!r}')