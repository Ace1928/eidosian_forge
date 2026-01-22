from __future__ import annotations
import numpy as np
import pytest
from xarray import DataArray, infer_freq
from xarray.coding.calendar_ops import convert_calendar, interp_calendar
from xarray.coding.cftime_offsets import date_range
from xarray.testing import assert_identical
from xarray.tests import requires_cftime
@pytest.mark.parametrize('source, target, use_cftime, freq', [('standard', 'noleap', None, 'D'), ('noleap', 'proleptic_gregorian', True, 'D'), ('noleap', 'all_leap', None, 'D'), ('all_leap', 'proleptic_gregorian', False, '4h')])
def test_convert_calendar(source, target, use_cftime, freq):
    src = DataArray(date_range('2004-01-01', '2004-12-31', freq=freq, calendar=source), dims=('time',), name='time')
    da_src = DataArray(np.linspace(0, 1, src.size), dims=('time',), coords={'time': src})
    conv = convert_calendar(da_src, target, use_cftime=use_cftime)
    assert conv.time.dt.calendar == target
    if source != 'noleap':
        expected_times = date_range('2004-01-01', '2004-12-31', freq=freq, use_cftime=use_cftime, calendar=target)
    else:
        expected_times_pre_leap = date_range('2004-01-01', '2004-02-28', freq=freq, use_cftime=use_cftime, calendar=target)
        expected_times_post_leap = date_range('2004-03-01', '2004-12-31', freq=freq, use_cftime=use_cftime, calendar=target)
        expected_times = expected_times_pre_leap.append(expected_times_post_leap)
    np.testing.assert_array_equal(conv.time, expected_times)