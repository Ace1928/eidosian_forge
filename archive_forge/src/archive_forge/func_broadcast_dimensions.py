from __future__ import annotations
import itertools
import os
from collections.abc import Hashable, Iterable, Mapping, Sequence
from itertools import product
from math import prod
from typing import Any
import tlz as toolz
import dask
from dask.base import clone_key, get_name_from_key, tokenize
from dask.core import flatten, ishashable, keys_in_tasks, reverse_dict
from dask.highlevelgraph import HighLevelGraph, Layer
from dask.optimization import SubgraphCallable, fuse
from dask.typing import Graph, Key
from dask.utils import (
def broadcast_dimensions(argpairs, numblocks, sentinels=(1, (1,)), consolidate=None):
    """Find block dimensions from arguments

    Parameters
    ----------
    argpairs : iterable
        name, ijk index pairs
    numblocks : dict
        maps {name: number of blocks}
    sentinels : iterable (optional)
        values for singleton dimensions
    consolidate : func (optional)
        use this to reduce each set of common blocks into a smaller set

    Examples
    --------
    >>> argpairs = [('x', 'ij'), ('y', 'ji')]
    >>> numblocks = {'x': (2, 3), 'y': (3, 2)}
    >>> broadcast_dimensions(argpairs, numblocks)
    {'i': 2, 'j': 3}

    Supports numpy broadcasting rules

    >>> argpairs = [('x', 'ij'), ('y', 'ij')]
    >>> numblocks = {'x': (2, 1), 'y': (1, 3)}
    >>> broadcast_dimensions(argpairs, numblocks)
    {'i': 2, 'j': 3}

    Works in other contexts too

    >>> argpairs = [('x', 'ij'), ('y', 'ij')]
    >>> d = {'x': ('Hello', 1), 'y': (1, (2, 3))}
    >>> broadcast_dimensions(argpairs, d)
    {'i': 'Hello', 'j': (2, 3)}
    """
    argpairs2 = [(a, ind) for a, ind in argpairs if ind is not None]
    L = toolz.concat([zip(inds, dims) for (x, inds), (x, dims) in toolz.join(toolz.first, argpairs2, toolz.first, numblocks.items())])
    g = toolz.groupby(0, L)
    g = {k: {d for i, d in v} for k, v in g.items()}
    g2 = {k: v - set(sentinels) if len(v) > 1 else v for k, v in g.items()}
    if consolidate:
        return toolz.valmap(consolidate, g2)
    if g2 and (not set(map(len, g2.values())) == {1}):
        raise ValueError('Shapes do not align %s' % g)
    return toolz.valmap(toolz.first, g2)