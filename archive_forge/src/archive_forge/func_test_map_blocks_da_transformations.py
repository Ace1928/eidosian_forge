from __future__ import annotations
import operator
import pickle
import sys
from contextlib import suppress
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, Variable
from xarray.core import duck_array_ops
from xarray.core.duck_array_ops import lazy_array_equiv
from xarray.testing import assert_chunks_equal
from xarray.tests import (
from xarray.tests.test_backends import create_tmp_file
@pytest.mark.parametrize('func', [lambda x: x, lambda x: x.to_dataset(), lambda x: x.drop_vars('x'), lambda x: x.expand_dims(k=[1, 2, 3]), lambda x: x.expand_dims(k=3), lambda x: x.assign_coords(new_coord=('y', x.y.data * 2)), lambda x: x.astype(np.int32), lambda x: x.x])
def test_map_blocks_da_transformations(func, map_da):
    with raise_if_dask_computes():
        actual = xr.map_blocks(func, map_da)
    assert_identical(actual, func(map_da))