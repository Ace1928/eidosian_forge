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
def test_date_range_errors() -> None:
    with pytest.raises(ValueError, match='Date range is invalid'):
        date_range('1400-01-01', periods=1, freq='D', calendar='standard', use_cftime=False)
    with pytest.raises(ValueError, match='Date range is invalid'):
        date_range('2480-01-01', periods=1, freq='D', calendar='proleptic_gregorian', use_cftime=False)
    with pytest.raises(ValueError, match='Invalid calendar '):
        date_range('1900-01-01', periods=1, freq='D', calendar='noleap', use_cftime=False)