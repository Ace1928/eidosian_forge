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
def bag_zip(*bags):
    """Partition-wise bag zip

    All passed bags must have the same number of partitions.

    NOTE: corresponding partitions should have the same length; if they do not,
    the "extra" elements from the longer partition(s) will be dropped.  If you
    have this case chances are that what you really need is a data alignment
    mechanism like pandas's, and not a missing value filler like zip_longest.

    Examples
    --------

    Correct usage:

    >>> import dask.bag as db
    >>> evens = db.from_sequence(range(0, 10, 2), partition_size=4)
    >>> odds = db.from_sequence(range(1, 10, 2), partition_size=4)
    >>> pairs = db.zip(evens, odds)
    >>> list(pairs)
    [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

    Incorrect usage:

    >>> numbers = db.range(31, npartitions=1)
    >>> fizz = numbers.filter(lambda n: n % 3 == 0)
    >>> buzz = numbers.filter(lambda n: n % 5 == 0)
    >>> fizzbuzz = db.zip(fizz, buzz)
    >>> list(fizzbuzz)
    [(0, 0), (3, 5), (6, 10), (9, 15), (12, 20), (15, 25), (18, 30)]

    When what you really wanted was more along the lines of the following:

    >>> list(fizzbuzz) # doctest: +SKIP
    (0, 0), (3, None), (None, 5), (6, None), (9, None), (None, 10),
    (12, None), (15, 15), (18, None), (None, 20),
    (21, None), (24, None), (None, 25), (27, None), (30, 30)
    """
    npartitions = bags[0].npartitions
    assert all((bag.npartitions == npartitions for bag in bags))
    name = 'zip-' + tokenize(*bags)
    dsk = {(name, i): (reify, (zip,) + tuple(((bag.name, i) for bag in bags))) for i in range(npartitions)}
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=bags)
    return Bag(graph, name, npartitions)