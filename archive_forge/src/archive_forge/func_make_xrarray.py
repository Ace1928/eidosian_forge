from __future__ import annotations
import math
import pickle
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Variable
from xarray.namedarray.pycompat import array_type
from xarray.tests import assert_equal, assert_identical, requires_dask
def make_xrarray(dim_lengths, coords=None, name='test'):
    if coords is None:
        coords = {d: np.arange(n) for d, n in dim_lengths.items()}
    return xr.DataArray(make_sparray(shape=tuple(dim_lengths.values())), dims=tuple(coords.keys()), coords=coords, name=name)