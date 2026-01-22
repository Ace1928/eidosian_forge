from __future__ import annotations
import numpy as np
import pandas as pd
import dask
import dask.array as da
import dask.dataframe as dd
from dask.base import tokenize, compute
from datashader.core import bypixel
from datashader.utils import apply
from datashader.compiler import compile_components
from datashader.glyphs import Glyph, LineAxis0
from datashader.utils import Dispatcher
def wrapped_combine(x, axis, keepdims):
    """ wrap datashader combine in dask.array.reduction combine """
    if isinstance(x, list):
        return combine(x)
    elif isinstance(x, tuple):
        return x
    else:
        raise TypeError('Unknown type %s in wrapped_combine' % type(x))