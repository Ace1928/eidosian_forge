from __future__ import annotations
import contextlib
import math
import operator
import os
import pickle
import re
import sys
import traceback
import uuid
import warnings
from bisect import bisect
from collections.abc import (
from functools import partial, reduce, wraps
from itertools import product, zip_longest
from numbers import Integral, Number
from operator import add, mul
from threading import Lock
from typing import Any, TypeVar, Union, cast
import numpy as np
from numpy.typing import ArrayLike
from tlz import accumulate, concat, first, frequencies, groupby, partition
from tlz.curried import pluck
from dask import compute, config, core
from dask.array import chunk
from dask.array.chunk import getitem
from dask.array.chunk_types import is_valid_array_chunk, is_valid_chunk_type
from dask.array.dispatch import (  # noqa: F401
from dask.array.numpy_compat import _Recurser
from dask.array.slicing import replace_ellipsis, setitem_array, slice_array
from dask.base import (
from dask.blockwise import blockwise as core_blockwise
from dask.blockwise import broadcast_dimensions
from dask.context import globalmethod
from dask.core import quote
from dask.delayed import Delayed, delayed
from dask.highlevelgraph import HighLevelGraph, MaterializedLayer
from dask.layers import ArraySliceDep, reshapelist
from dask.sizeof import sizeof
from dask.typing import Graph, Key, NestedKeys
from dask.utils import (
from dask.widgets import get_template
from dask.array.optimization import fuse_slice, optimize
from dask.array.blockwise import blockwise
from dask.array.utils import compute_meta, meta_from_array
def graph_from_arraylike(arr, chunks, shape, name, getitem=getter, lock=False, asarray=True, dtype=None, inline_array=False) -> HighLevelGraph:
    """
    HighLevelGraph for slicing chunks from an array-like according to a chunk pattern.

    If ``inline_array`` is True, this make a Blockwise layer of slicing tasks where the
    array-like is embedded into every task.,

    If ``inline_array`` is False, this inserts the array-like as a standalone value in
    a MaterializedLayer, then generates a Blockwise layer of slicing tasks that refer
    to it.

    >>> dict(graph_from_arraylike(arr, chunks=(2, 3), shape=(4, 6), name="X", inline_array=True))  # doctest: +SKIP
    {(arr, 0, 0): (getter, arr, (slice(0, 2), slice(0, 3))),
     (arr, 1, 0): (getter, arr, (slice(2, 4), slice(0, 3))),
     (arr, 1, 1): (getter, arr, (slice(2, 4), slice(3, 6))),
     (arr, 0, 1): (getter, arr, (slice(0, 2), slice(3, 6)))}

    >>> dict(  # doctest: +SKIP
            graph_from_arraylike(arr, chunks=((2, 2), (3, 3)), shape=(4,6), name="X", inline_array=False)
        )
    {"original-X": arr,
     ('X', 0, 0): (getter, 'original-X', (slice(0, 2), slice(0, 3))),
     ('X', 1, 0): (getter, 'original-X', (slice(2, 4), slice(0, 3))),
     ('X', 1, 1): (getter, 'original-X', (slice(2, 4), slice(3, 6))),
     ('X', 0, 1): (getter, 'original-X', (slice(0, 2), slice(3, 6)))}
    """
    chunks = normalize_chunks(chunks, shape, dtype=dtype)
    out_ind = tuple(range(len(shape)))
    if has_keyword(getitem, 'asarray') and has_keyword(getitem, 'lock') and (not asarray or lock):
        kwargs = {'asarray': asarray, 'lock': lock}
    else:
        kwargs = {}
    if inline_array:
        layer = core_blockwise(getitem, name, out_ind, arr, None, ArraySliceDep(chunks), out_ind, numblocks={}, **kwargs)
        return HighLevelGraph.from_collections(name, layer)
    else:
        original_name = 'original-' + name
        layers = {}
        layers[original_name] = MaterializedLayer({original_name: arr})
        layers[name] = core_blockwise(getitem, name, out_ind, original_name, None, ArraySliceDep(chunks), out_ind, numblocks={}, **kwargs)
        deps = {original_name: set(), name: {original_name}}
        return HighLevelGraph(layers, deps)