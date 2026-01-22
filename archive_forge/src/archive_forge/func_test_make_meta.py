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
def test_make_meta(map_ds):
    from xarray.core.parallel import make_meta
    meta = make_meta(map_ds)
    for variable in map_ds._coord_names:
        assert variable in meta._coord_names
        assert meta.coords[variable].shape == (0,) * meta.coords[variable].ndim
    for variable in map_ds.data_vars:
        assert variable in meta.data_vars
        assert meta.data_vars[variable].shape == (0,) * meta.data_vars[variable].ndim