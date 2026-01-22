from __future__ import annotations
import pickle
from datetime import timedelta
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.coding.cftimeindex import (
from xarray.tests import (
from xarray.tests.test_coding_times import (
@requires_cftime
@pytest.mark.parametrize('method', ['floor', 'ceil', 'round'])
def test_rounding_methods_empty_cftimindex(method):
    index = CFTimeIndex([])
    result = getattr(index, method)('2s')
    expected = CFTimeIndex([])
    assert result.equals(expected)
    assert result is not index