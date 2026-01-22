from unittest.mock import Mock
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import (
@pytest.mark.parametrize('direction_label', [False, True])
@pytest.mark.parametrize('test_ticks,expected', [pytest.param([-3.75, -3.5], ['3°45′W', '3°30′W'], id='minutes_no_hide'), pytest.param([-3.5, -3], ['30′', '3°W'], id='minutes_hide'), pytest.param([-3 - 2 * ONE_MIN - 30 * ONE_SEC], ['3°2′30″W'], id='seconds')])
def test_LongitudeFormatter_minutes_seconds(test_ticks, direction_label, expected):
    formatter = LongitudeFormatter(dms=True, auto_hide=True, direction_label=direction_label)
    formatter.set_locs(test_ticks)
    result = [formatter(tick) for tick in test_ticks]
    prefix = '' if direction_label else '-'
    suffix = 'W' if direction_label else ''
    expected = [f'{prefix}{text[:-1]}{suffix}' if text[-1] == 'W' else text for text in expected]
    assert result == expected