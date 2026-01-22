from __future__ import annotations
from itertools import product
from typing import Callable, Literal
import numpy as np
import pandas as pd
import pytest
from xarray import CFTimeIndex
from xarray.coding.cftime_offsets import (
from xarray.coding.frequencies import infer_freq
from xarray.core.dataarray import DataArray
from xarray.tests import (
@pytest.mark.parametrize(('offset', 'expected_date_args'), _ADD_TESTS, ids=_id_func)
def test_add_sub_monthly(offset, expected_date_args, calendar):
    date_type = get_date_type(calendar)
    initial = date_type(1, 1, 1)
    expected = date_type(*expected_date_args)
    result = offset + initial
    assert result == expected