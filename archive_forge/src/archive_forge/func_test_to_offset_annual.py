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
@pytest.mark.parametrize(('month_int', 'month_label'), list(_MONTH_ABBREVIATIONS.items()) + [(0, '')])
@pytest.mark.parametrize('multiple', [None, 2, -1])
@pytest.mark.parametrize('offset_str', ['AS', 'A', 'YS', 'Y'])
@pytest.mark.filterwarnings('ignore::FutureWarning')
def test_to_offset_annual(month_label, month_int, multiple, offset_str):
    freq = offset_str
    offset_type = _ANNUAL_OFFSET_TYPES[offset_str]
    if month_label:
        freq = '-'.join([freq, month_label])
    if multiple:
        freq = f'{multiple}{freq}'
    result = to_offset(freq)
    if multiple and month_int:
        expected = offset_type(n=multiple, month=month_int)
    elif multiple:
        expected = offset_type(n=multiple)
    elif month_int:
        expected = offset_type(month=month_int)
    else:
        expected = offset_type()
    assert result == expected