from __future__ import annotations
import pickle
from datetime import timedelta
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.coding.cftimeindex import (
from xarray.tests import (
from xarray.tests.test_coding_times import (
@requires_cftime
@pytest.mark.parametrize(('field', 'expected'), [('year', [1, 1, 2, 2]), ('month', [1, 2, 1, 2]), ('day', [1, 1, 1, 1]), ('hour', [0, 0, 0, 0]), ('minute', [0, 0, 0, 0]), ('second', [0, 0, 0, 0]), ('microsecond', [0, 0, 0, 0])])
def test_cftimeindex_field_accessors(index, field, expected):
    result = getattr(index, field)
    expected = np.array(expected, dtype=np.int64)
    assert_array_equal(result, expected)
    assert result.dtype == expected.dtype