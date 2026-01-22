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
def test_map_blocks_hlg_layers():
    ds = xr.Dataset({'x': (('a',), dask.array.ones(10, chunks=(5,))), 'z': (('b',), dask.array.ones(10, chunks=(5,)))})
    mapped = ds.map_blocks(lambda x: x)
    xr.testing.assert_equal(mapped, ds)