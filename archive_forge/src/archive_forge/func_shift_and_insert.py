from __future__ import annotations
import os
import re
from inspect import getmro
import numba as nb
import numpy as np
import pandas as pd
from toolz import memoize
from xarray import DataArray
import dask.dataframe as dd
import datashader.datashape as datashape
@ngjit
def shift_and_insert(target, value, index):
    """Insert a value into a 1D array at a particular index, but before doing
    that shift the previous values along one to make room. For use in
    ``FloatingNReduction`` classes such as ``max_n`` and ``first_n`` which
    store ``n`` values per pixel.

    Parameters
    ----------
    target : 1d numpy array
        Target pixel array.

    value : float
        Value to insert into target pixel array.

    index : int
        Index to insert at.

    Returns
    -------
    Index beyond insertion, i.e. where the first shifted value now sits.
    """
    n = len(target)
    for i in range(n - 1, index, -1):
        target[i] = target[i - 1]
    target[index] = value
    return index + 1