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
def groupby_disk(b, grouper, npartitions=None, blocksize=2 ** 20):
    if npartitions is None:
        npartitions = b.npartitions
    token = tokenize(b, grouper, npartitions, blocksize)
    import partd
    p = ('partd-' + token,)
    dirname = config.get('temporary_directory', None)
    if dirname:
        file = (apply, partd.File, (), {'dir': dirname})
    else:
        file = (partd.File,)
    try:
        dsk1 = {p: (partd.Python, (partd.Snappy, file))}
    except AttributeError:
        dsk1 = {p: (partd.Python, file)}
    name = f'groupby-part-{funcname(grouper)}-{token}'
    dsk2 = {(name, i): (partition, grouper, (b.name, i), npartitions, p, blocksize) for i in range(b.npartitions)}
    barrier_token = 'groupby-barrier-' + token
    dsk3 = {barrier_token: (chunk.barrier,) + tuple(dsk2)}
    name = 'groupby-collect-' + token
    dsk4 = {(name, i): (collect, grouper, i, p, barrier_token) for i in range(npartitions)}
    dsk = merge(dsk1, dsk2, dsk3, dsk4)
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=[b])
    return type(b)(graph, name, npartitions)