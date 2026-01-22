from __future__ import annotations
import warnings
from abc import ABC
from copy import copy, deepcopy
from datetime import datetime, timedelta
from textwrap import dedent
from typing import Generic
import numpy as np
import pandas as pd
import pytest
import pytz
from xarray import DataArray, Dataset, IndexVariable, Variable, set_options
from xarray.core import dtypes, duck_array_ops, indexing
from xarray.core.common import full_like, ones_like, zeros_like
from xarray.core.indexing import (
from xarray.core.types import T_DuckArray
from xarray.core.utils import NDArrayMixin
from xarray.core.variable import as_compatible_data, as_variable
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
from xarray.tests.test_namedarray import NamedArraySubclassobjects
def test_getitem_basic(self):
    v = self.cls(['x', 'y'], [[0, 1, 2], [3, 4, 5]])
    v_new = v[0]
    assert v_new.dims == ('y',)
    assert_array_equal(v_new, v._data[0])
    v_new = v[:2]
    assert v_new.dims == ('x', 'y')
    assert_array_equal(v_new, v._data[:2])
    v_new = v[[0]]
    assert v_new.dims == ('x', 'y')
    assert_array_equal(v_new, v._data[[0]])
    v_new = v[[]]
    assert v_new.dims == ('x', 'y')
    assert_array_equal(v_new, v._data[[]])
    v_new = v[dict(x=0)]
    assert v_new.dims == ('y',)
    assert_array_equal(v_new, v._data[0])
    v_new = v[dict(x=0, y=slice(None))]
    assert v_new.dims == ('y',)
    assert_array_equal(v_new, v._data[0])
    v_new = v[dict(x=0, y=1)]
    assert v_new.dims == ()
    assert_array_equal(v_new, v._data[0, 1])
    v_new = v[dict(y=1)]
    assert v_new.dims == ('x',)
    assert_array_equal(v_new, v._data[:, 1])
    v_new = v[slice(None), 1]
    assert v_new.dims == ('x',)
    assert_array_equal(v_new, v._data[:, 1])
    v_new = v[0, 0]
    v_new[...] += 99
    assert_array_equal(v_new, v._data[0, 0])