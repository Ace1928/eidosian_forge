from __future__ import annotations
import math
import numbers
from collections.abc import Iterable
from enum import Enum
from typing import Any
from dask import config, core, utils
from dask.base import normalize_token, tokenize
from dask.core import (
from dask.typing import Graph, Key
def functions_of(task):
    """Set of functions contained within nested task

    Examples
    --------
    >>> inc = lambda x: x + 1
    >>> add = lambda x, y: x + y
    >>> mul = lambda x, y: x * y
    >>> task = (add, (mul, 1, 2), (inc, 3))  # doctest: +SKIP
    >>> functions_of(task)  # doctest: +SKIP
    set([add, mul, inc])
    """
    funcs = set()
    work = [task]
    sequence_types = {list, tuple}
    while work:
        new_work = []
        for task in work:
            if type(task) in sequence_types:
                if istask(task):
                    funcs.add(unwrap_partial(task[0]))
                    new_work.extend(task[1:])
                else:
                    new_work.extend(task)
        work = new_work
    return funcs