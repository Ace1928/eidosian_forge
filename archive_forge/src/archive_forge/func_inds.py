from __future__ import annotations
import functools
import math
import operator
from collections import defaultdict
from collections.abc import Callable
from itertools import product
from typing import Any
import tlz as toolz
from tlz.curried import map
from dask.base import tokenize
from dask.blockwise import Blockwise, BlockwiseDep, BlockwiseDepDict, blockwise_token
from dask.core import flatten
from dask.highlevelgraph import Layer
from dask.utils import apply, cached_cumsum, concrete, insert
def inds(i, ind):
    rv = []
    if ind - 0.9 > 0:
        rv.append(ind - 0.9)
    rv.append(ind)
    if ind + 0.9 < dims[i] - 1:
        rv.append(ind + 0.9)
    return rv