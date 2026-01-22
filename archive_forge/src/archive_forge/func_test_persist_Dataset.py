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
@pytest.mark.parametrize('persist', [lambda x: x.persist(), lambda x: dask.persist(x)[0]])
def test_persist_Dataset(persist):
    ds = Dataset({'foo': ('x', range(5)), 'bar': ('x', range(5))}).chunk()
    ds = ds + 1
    n = len(ds.foo.data.dask)
    ds2 = persist(ds)
    assert len(ds2.foo.data.dask) == 1
    assert len(ds.foo.data.dask) == n