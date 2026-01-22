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
def test_get_slice_bound(date_type, index):
    result = index.get_slice_bound('0001', 'left')
    expected = 0
    assert result == expected
    result = index.get_slice_bound('0001', 'right')
    expected = 2
    assert result == expected
    result = index.get_slice_bound(date_type(1, 3, 1), 'left')
    expected = 2
    assert result == expected
    result = index.get_slice_bound(date_type(1, 3, 1), 'right')
    expected = 2
    assert result == expected