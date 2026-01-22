from __future__ import annotations
import heapq
import math
from functools import reduce
from itertools import chain, count, product
from operator import add, itemgetter, mul
from warnings import warn
import numpy as np
import tlz as toolz
from tlz import accumulate
from dask import config
from dask.array.chunk import getitem
from dask.array.core import Array, concatenate3, normalize_chunks
from dask.array.utils import validate_axis
from dask.array.wrap import empty
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
from dask.utils import parse_bytes
def format_blocks(blocks):
    """
    Pretty-format *blocks*.

    >>> format_blocks((10, 10, 10))
    3*[10]
    >>> format_blocks((2, 3, 4))
    [2, 3, 4]
    >>> format_blocks((10, 10, 5, 6, 2, 2, 2, 7))
    2*[10] | [5, 6] | 3*[2] | [7]
    """
    assert isinstance(blocks, tuple) and all((isinstance(x, int) or math.isnan(x) for x in blocks))
    return _PrettyBlocks(blocks)