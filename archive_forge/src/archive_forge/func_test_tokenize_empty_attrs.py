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
@pytest.mark.parametrize('obj', [make_ds(), make_ds().variables['c2'], make_ds().variables['x']])
def test_tokenize_empty_attrs(obj):
    """Issues #6970 and #8788"""
    obj.attrs = {}
    assert obj._attrs is None
    a = dask.base.tokenize(obj)
    assert obj.attrs == {}
    assert obj._attrs == {}
    b = dask.base.tokenize(obj)
    assert a == b
    obj2 = obj.copy()
    c = dask.base.tokenize(obj2)
    assert a == c