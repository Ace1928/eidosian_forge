from __future__ import annotations
import collections
import functools
import logging
import math
import os
import threading
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from typing import (
def register_cache(cls: type[BaseCache], clobber: bool=False) -> None:
    """'Register' cache implementation.

    Parameters
    ----------
    clobber: bool, optional
        If set to True (default is False) - allow to overwrite existing
        entry.

    Raises
    ------
    ValueError
    """
    name = cls.name
    if not clobber and name in caches:
        raise ValueError(f'Cache with name {name!r} is already known: {caches[name]}')
    caches[name] = cls