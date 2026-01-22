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
@pytest.mark.parametrize(['keep_attrs', 'attrs', 'expected'], [pytest.param(None, {'a': 1, 'b': 2}, {'a': 1, 'b': 2}, id='default'), pytest.param(False, {'a': 1, 'b': 2}, {}, id='False'), pytest.param(True, {'a': 1, 'b': 2}, {'a': 1, 'b': 2}, id='True')])
def test_pad_keep_attrs(self, keep_attrs, attrs, expected):
    data = np.arange(10, dtype=float)
    v = self.cls(['x'], data, attrs)
    keep_attrs_ = 'default' if keep_attrs is None else keep_attrs
    with set_options(keep_attrs=keep_attrs_):
        actual = v.pad({'x': (1, 1)}, mode='constant', constant_values=np.nan)
        assert actual.attrs == expected
    actual = v.pad({'x': (1, 1)}, mode='constant', constant_values=np.nan, keep_attrs=keep_attrs)
    assert actual.attrs == expected