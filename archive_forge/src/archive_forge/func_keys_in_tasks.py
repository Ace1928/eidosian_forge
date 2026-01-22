from __future__ import annotations
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping
from typing import Any, Literal, TypeVar, cast, overload
from dask.typing import Graph, Key, NoDefault, no_default
def keys_in_tasks(keys: Collection[Key], tasks: Iterable[Any], as_list: bool=False):
    """Returns the keys in `keys` that are also in `tasks`

    Examples
    --------
    >>> inc = lambda x: x + 1
    >>> add = lambda x, y: x + y
    >>> dsk = {'x': 1,
    ...        'y': (inc, 'x'),
    ...        'z': (add, 'x', 'y'),
    ...        'w': (inc, 'z'),
    ...        'a': (add, (inc, 'x'), 1)}

    >>> keys_in_tasks(dsk, ['x', 'y', 'j'])  # doctest: +SKIP
    {'x', 'y'}
    """
    ret = []
    while tasks:
        work = []
        for w in tasks:
            typ = type(w)
            if typ is tuple and w and callable(w[0]):
                work.extend(w[1:])
            elif typ is list:
                work.extend(w)
            elif typ is dict:
                work.extend(w.values())
            else:
                try:
                    if w in keys:
                        ret.append(w)
                except TypeError:
                    pass
        tasks = work
    return ret if as_list else set(ret)