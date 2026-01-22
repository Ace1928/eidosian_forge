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
def test_broadcasting_math(self):
    x = np.random.randn(2, 3)
    v = Variable(['a', 'b'], x)
    assert_identical(v * v, Variable(['a', 'b'], np.einsum('ab,ab->ab', x, x)))
    assert_identical(v * v[0], Variable(['a', 'b'], np.einsum('ab,b->ab', x, x[0])))
    assert_identical(v[0] * v, Variable(['b', 'a'], np.einsum('b,ab->ba', x[0], x)))
    assert_identical(v[0] * v[:, 0], Variable(['b', 'a'], np.einsum('b,a->ba', x[0], x[:, 0])))
    y = np.random.randn(3, 4, 5)
    w = Variable(['b', 'c', 'd'], y)
    assert_identical(v * w, Variable(['a', 'b', 'c', 'd'], np.einsum('ab,bcd->abcd', x, y)))
    assert_identical(w * v, Variable(['b', 'c', 'd', 'a'], np.einsum('bcd,ab->bcda', y, x)))
    assert_identical(v * w[0], Variable(['a', 'b', 'c', 'd'], np.einsum('ab,cd->abcd', x, y[0])))