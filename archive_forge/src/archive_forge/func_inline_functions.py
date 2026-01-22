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
def inline_functions(dsk, output, fast_functions=None, inline_constants=False, dependencies=None):
    """Inline cheap functions into larger operations

    Examples
    --------
    >>> inc = lambda x: x + 1
    >>> add = lambda x, y: x + y
    >>> double = lambda x: x * 2
    >>> dsk = {'out': (add, 'i', 'd'),  # doctest: +SKIP
    ...        'i': (inc, 'x'),
    ...        'd': (double, 'y'),
    ...        'x': 1, 'y': 1}
    >>> inline_functions(dsk, [], [inc])  # doctest: +SKIP
    {'out': (add, (inc, 'x'), 'd'),
     'd': (double, 'y'),
     'x': 1, 'y': 1}

    Protect output keys.  In the example below ``i`` is not inlined because it
    is marked as an output key.

    >>> inline_functions(dsk, ['i', 'out'], [inc, double])  # doctest: +SKIP
    {'out': (add, 'i', (double, 'y')),
     'i': (inc, 'x'),
     'x': 1, 'y': 1}
    """
    if not fast_functions:
        return dsk
    output = set(output)
    fast_functions = set(fast_functions)
    if dependencies is None:
        dependencies = {k: get_dependencies(dsk, k) for k in dsk}
    dependents = reverse_dict(dependencies)

    def inlinable(v):
        try:
            return functions_of(v).issubset(fast_functions)
        except TypeError:
            return False
    keys = [k for k, v in dsk.items() if istask(v) and dependents[k] and (k not in output) and inlinable(v)]
    if keys:
        dsk = inline(dsk, keys, inline_constants=inline_constants, dependencies=dependencies)
        for k in keys:
            del dsk[k]
    return dsk