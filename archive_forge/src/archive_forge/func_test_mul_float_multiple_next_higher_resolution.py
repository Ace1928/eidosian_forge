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
def test_mul_float_multiple_next_higher_resolution():
    """Test more than one iteration through _next_higher_resolution is required."""
    assert 1e-06 * Second() == Microsecond()
    assert 1e-06 / 60 * Minute() == Microsecond()