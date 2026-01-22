from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core.missing import (
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
@requires_scipy
@pytest.mark.parametrize('method', ['barycentric', 'krogh', 'pchip', 'spline', 'akima'])
def test_scipy_methods_function(method) -> None:
    da, _ = make_interpolate_example_data((25, 25), 0.4, non_uniform=True)
    actual = da.interpolate_na(method=method, dim='time')
    assert (da.count('time') <= actual.count('time')).all()