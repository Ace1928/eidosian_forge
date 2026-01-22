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
@pytest.mark.parametrize(('offset', 'expected'), [(BaseCFTimeOffset(), '<BaseCFTimeOffset: n=1>'), (YearBegin(), '<YearBegin: n=1, month=1>'), (QuarterBegin(), '<QuarterBegin: n=1, month=3>')], ids=_id_func)
def test_str_and_repr(offset, expected):
    assert str(offset) == expected
    assert repr(offset) == expected