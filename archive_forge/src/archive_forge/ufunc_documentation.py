from __future__ import annotations
from functools import partial
from operator import getitem
import numpy as np
from dask import core
from dask.array.core import Array, apply_infer_dtype, asarray, blockwise, elemwise
from dask.base import is_dask_collection, normalize_token
from dask.highlevelgraph import HighLevelGraph
from dask.utils import derived_from, funcname
A serializable `frompyfunc` object