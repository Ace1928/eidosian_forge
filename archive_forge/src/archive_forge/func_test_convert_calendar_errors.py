from __future__ import annotations
import numpy as np
import pytest
from xarray import DataArray, infer_freq
from xarray.coding.calendar_ops import convert_calendar, interp_calendar
from xarray.coding.cftime_offsets import date_range
from xarray.testing import assert_identical
from xarray.tests import requires_cftime
@requires_cftime
def test_convert_calendar_errors():
    src_nl = DataArray(date_range('0000-01-01', '0000-12-31', freq='D', calendar='noleap'), dims=('time',), name='time')
    with pytest.raises(ValueError, match='Argument `align_on` must be specified'):
        convert_calendar(src_nl, '360_day')
    with pytest.raises(ValueError, match='Source time coordinate contains dates with year 0'):
        convert_calendar(src_nl, 'standard')
    src_360 = convert_calendar(src_nl, '360_day', align_on='year')
    with pytest.raises(ValueError, match='Argument `align_on` must be specified'):
        convert_calendar(src_360, 'noleap')
    da = DataArray([0, 1, 2], dims=('x',), name='x')
    with pytest.raises(ValueError, match='Coordinate x must contain datetime objects.'):
        convert_calendar(da, 'standard', dim='x')