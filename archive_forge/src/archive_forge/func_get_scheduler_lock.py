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
def get_scheduler_lock(collection=None, scheduler=None):
    """Get an instance of the appropriate lock for a certain situation based on
    scheduler used."""
    from dask import multiprocessing
    from dask.base import get_scheduler
    actual_get = get_scheduler(collections=[collection], scheduler=scheduler)
    if actual_get == multiprocessing.get:
        return multiprocessing.get_context().Manager().Lock()
    else:
        try:
            import distributed.lock
            from distributed.worker import get_client
            client = get_client()
        except (ImportError, ValueError):
            pass
        else:
            if actual_get == client.get:
                return distributed.lock.Lock()
    return SerializableLock()