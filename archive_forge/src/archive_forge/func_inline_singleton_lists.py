from __future__ import annotations
import io
import itertools
import math
import operator
import uuid
import warnings
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from functools import partial, reduce, wraps
from random import Random
from urllib.request import urlopen
import tlz as toolz
from fsspec.core import open_files
from tlz import (
from dask import config
from dask.bag import chunk
from dask.bag.avro import to_avro
from dask.base import (
from dask.blockwise import blockwise
from dask.context import globalmethod
from dask.core import flatten, get_dependencies, istask, quote, reverse_dict
from dask.delayed import Delayed, unpack_collections
from dask.highlevelgraph import HighLevelGraph
from dask.optimization import cull, fuse, inline
from dask.sizeof import sizeof
from dask.typing import Graph, NestedKeys, no_default
from dask.utils import (
def inline_singleton_lists(dsk, keys, dependencies=None):
    """Inline lists that are only used once.

    >>> d = {'b': (list, 'a'),
    ...      'c': (sum, 'b', 1)}
    >>> inline_singleton_lists(d, 'c')
    {'c': (<built-in function sum>, (<class 'list'>, 'a'), 1)}

    Pairs nicely with lazify afterwards.
    """
    if dependencies is None:
        dependencies = {k: get_dependencies(dsk, task=v) for k, v in dsk.items()}
    dependents = reverse_dict(dependencies)
    inline_keys = {k for k, v in dsk.items() if istask(v) and v and (v[0] is list) and (len(dependents[k]) == 1)}
    inline_keys.difference_update(flatten(keys))
    dsk = inline(dsk, inline_keys, inline_constants=False)
    for k in inline_keys:
        del dsk[k]
    return dsk