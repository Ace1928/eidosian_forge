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
def test_dataarray_getattr(self):
    data = build_dask_array('data')
    nonindex_coord = build_dask_array('coord')
    a = DataArray(data, dims=['x'], coords={'y': ('x', nonindex_coord)})
    with suppress(AttributeError):
        getattr(a, 'NOTEXIST')
    assert kernel_call_count == 0