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
def unpack_collections(*args, traverse=True):
    """Extract collections in preparation for compute/persist/etc...

    Intended use is to find all collections in a set of (possibly nested)
    python objects, do something to them (compute, etc...), then repackage them
    in equivalent python objects.

    Parameters
    ----------
    *args
        Any number of objects. If it is a dask collection, it's extracted and
        added to the list of collections returned. By default, python builtin
        collections are also traversed to look for dask collections (for more
        information see the ``traverse`` keyword).
    traverse : bool, optional
        If True (default), builtin python collections are traversed looking for
        any dask collections they might contain.

    Returns
    -------
    collections : list
        A list of all dask collections contained in ``args``
    repack : callable
        A function to call on the transformed collections to repackage them as
        they were in the original ``args``.
    """
    collections = []
    repack_dsk = {}
    collections_token = uuid.uuid4().hex

    def _unpack(expr):
        if is_dask_collection(expr):
            tok = tokenize(expr)
            if tok not in repack_dsk:
                repack_dsk[tok] = (getitem, collections_token, len(collections))
                collections.append(expr)
            return tok
        tok = uuid.uuid4().hex
        if not traverse:
            tsk = quote(expr)
        else:
            typ = list if isinstance(expr, Iterator) else type(expr)
            if typ in (list, tuple, set):
                tsk = (typ, [_unpack(i) for i in expr])
            elif typ in (dict, OrderedDict):
                tsk = (typ, [[_unpack(k), _unpack(v)] for k, v in expr.items()])
            elif dataclasses.is_dataclass(expr) and (not isinstance(expr, type)):
                tsk = (apply, typ, (), (dict, [[f.name, _unpack(getattr(expr, f.name))] for f in dataclasses.fields(expr)]))
            elif is_namedtuple_instance(expr):
                tsk = (typ, *[_unpack(i) for i in expr])
            else:
                return expr
        repack_dsk[tok] = tsk
        return tok
    out = uuid.uuid4().hex
    repack_dsk[out] = (tuple, [_unpack(i) for i in args])

    def repack(results):
        dsk = repack_dsk.copy()
        dsk[collections_token] = quote(results)
        return simple_get(dsk, out)
    return (collections, repack)