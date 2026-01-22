from __future__ import annotations
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping
from typing import Any, Literal, TypeVar, cast, overload
from dask.typing import Graph, Key, NoDefault, no_default
def ishashable(x):
    """Is x hashable?

    Examples
    --------

    >>> ishashable(1)
    True
    >>> ishashable([1])
    False

    See Also
    --------
    iskey
    """
    try:
        hash(x)
        return True
    except TypeError:
        return False