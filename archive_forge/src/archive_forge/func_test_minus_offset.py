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
@pytest.mark.parametrize(('a', 'b'), zip(_EQ_TESTS_A, _EQ_TESTS_B), ids=_id_func)
def test_minus_offset(a, b):
    result = b - a
    expected = a
    assert result == expected