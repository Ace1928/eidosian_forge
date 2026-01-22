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
@pytest.mark.skipif(has_pandas_ge_2_2, reason='only relevant for pandas lt 2.2')
@pytest.mark.parametrize('year_alias', ('YE', 'Y', 'A'))
@pytest.mark.parametrize('n', ('', '2'))
def test_legacy_to_new_freq_anchored(year_alias, n):
    for month in _MONTH_ABBREVIATIONS.values():
        freq = f'{n}{year_alias}-{month}'
        result = _legacy_to_new_freq(freq)
        expected = f'{n}YE-{month}'
        assert result == expected