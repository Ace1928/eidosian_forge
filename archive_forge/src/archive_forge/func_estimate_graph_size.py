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
def estimate_graph_size(old_chunks, new_chunks):
    """Estimate the graph size during a rechunk computation."""
    crossed_size = reduce(mul, (len(oc) + len(nc) - 1 if oc != nc else len(oc) for oc, nc in zip(old_chunks, new_chunks)))
    return crossed_size