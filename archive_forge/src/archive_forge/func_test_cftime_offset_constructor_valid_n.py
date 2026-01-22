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
@pytest.mark.parametrize(('offset', 'expected_n'), [(BaseCFTimeOffset(), 1), (YearBegin(), 1), (YearEnd(), 1), (QuarterBegin(), 1), (QuarterEnd(), 1), (Tick(), 1), (Day(), 1), (Hour(), 1), (Minute(), 1), (Second(), 1), (Millisecond(), 1), (Microsecond(), 1), (BaseCFTimeOffset(n=2), 2), (YearBegin(n=2), 2), (YearEnd(n=2), 2), (QuarterBegin(n=2), 2), (QuarterEnd(n=2), 2), (Tick(n=2), 2), (Day(n=2), 2), (Hour(n=2), 2), (Minute(n=2), 2), (Second(n=2), 2), (Millisecond(n=2), 2), (Microsecond(n=2), 2)], ids=_id_func)
def test_cftime_offset_constructor_valid_n(offset, expected_n):
    assert offset.n == expected_n