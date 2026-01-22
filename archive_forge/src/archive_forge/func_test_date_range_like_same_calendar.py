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
def test_date_range_like_same_calendar():
    src = date_range('2000-01-01', periods=12, freq='6h', use_cftime=False)
    out = date_range_like(src, 'standard', use_cftime=False)
    assert src is out