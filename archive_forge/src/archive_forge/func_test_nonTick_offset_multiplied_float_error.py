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
@pytest.mark.parametrize('offset', [YearBegin(), YearEnd(), QuarterBegin(), QuarterEnd(), MonthBegin(), MonthEnd()], ids=_id_func)
def test_nonTick_offset_multiplied_float_error(offset):
    """Test that the appropriate error is raised if a non-Tick offset is
    multiplied by a float."""
    with pytest.raises(TypeError, match='unsupported operand type'):
        offset * 0.5