from __future__ import annotations
import numbers
import warnings
import tlz as toolz
from dask import base, utils
from dask.blockwise import blockwise as core_blockwise
from dask.delayed import unpack_collections
from dask.highlevelgraph import HighLevelGraph
from dask.array.core import new_da_object
def atop(*args, **kwargs):
    warnings.warn('The da.atop function has moved to da.blockwise')
    return blockwise(*args, **kwargs)