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
@pytest.mark.parametrize(('a', 'b'), list(zip(np.roll(_EQ_TESTS_A, 1), _EQ_TESTS_B)) + [(YearEnd(month=1), YearEnd(month=2))], ids=_id_func)
def test_minus_offset_error(a, b):
    with pytest.raises(TypeError):
        b - a