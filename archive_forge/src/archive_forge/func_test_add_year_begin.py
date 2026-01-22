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
@pytest.mark.parametrize(('initial_date_args', 'offset', 'expected_date_args'), [((1, 1, 1), YearBegin(), (2, 1, 1)), ((1, 1, 1), YearBegin(n=2), (3, 1, 1)), ((1, 1, 1), YearBegin(month=2), (1, 2, 1)), ((1, 1, 7), YearBegin(n=2), (3, 1, 1)), ((2, 2, 1), YearBegin(n=-1), (2, 1, 1)), ((1, 1, 2), YearBegin(n=-1), (1, 1, 1)), ((1, 1, 1, 5, 5, 5, 5), YearBegin(), (2, 1, 1, 5, 5, 5, 5)), ((2, 1, 1, 5, 5, 5, 5), YearBegin(n=-1), (1, 1, 1, 5, 5, 5, 5))], ids=_id_func)
def test_add_year_begin(calendar, initial_date_args, offset, expected_date_args):
    date_type = get_date_type(calendar)
    initial = date_type(*initial_date_args)
    result = initial + offset
    expected = date_type(*expected_date_args)
    assert result == expected