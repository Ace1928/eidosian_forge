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
@pytest.mark.parametrize('offset_str', ['QS', 'Q', 'QE'])
@pytest.mark.filterwarnings('ignore::FutureWarning')
def test_to_offset_quarter(month_label, month_int, multiple, offset_str):
    freq = offset_str
    offset_type = _QUARTER_OFFSET_TYPES[offset_str]
    if month_label:
        freq = '-'.join([freq, month_label])
    if multiple:
        freq = f'{multiple}{freq}'
    result = to_offset(freq)
    if multiple and month_int:
        expected = offset_type(n=multiple, month=month_int)
    elif multiple:
        if month_int:
            expected = offset_type(n=multiple)
        elif offset_type == QuarterBegin:
            expected = offset_type(n=multiple, month=1)
        elif offset_type == QuarterEnd:
            expected = offset_type(n=multiple, month=12)
    elif month_int:
        expected = offset_type(month=month_int)
    elif offset_type == QuarterBegin:
        expected = offset_type(month=1)
    elif offset_type == QuarterEnd:
        expected = offset_type(month=12)
    assert result == expected