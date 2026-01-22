from __future__ import annotations
import functools
import operator
import pickle
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import xarray as xr
from xarray.core.alignment import broadcast
from xarray.core.computation import (
from xarray.tests import (
def stack_negative(obj):

    def func(x):
        return np.stack([x, -x], axis=-1)
    return apply_ufunc(func, obj, output_core_dims=[['sign']], dask='parallelized', output_dtypes=[obj.dtype], dask_gufunc_kwargs=dict(output_sizes={'sign': 2}))