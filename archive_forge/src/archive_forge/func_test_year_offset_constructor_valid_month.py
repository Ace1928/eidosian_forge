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
@pytest.mark.parametrize(('offset', 'expected_month'), [(YearBegin(), 1), (YearEnd(), 12), (YearBegin(month=5), 5), (YearEnd(month=5), 5), (QuarterBegin(), 3), (QuarterEnd(), 3), (QuarterBegin(month=5), 5), (QuarterEnd(month=5), 5)], ids=_id_func)
def test_year_offset_constructor_valid_month(offset, expected_month):
    assert offset.month == expected_month