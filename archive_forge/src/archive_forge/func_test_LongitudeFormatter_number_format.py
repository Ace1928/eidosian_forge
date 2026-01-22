from unittest.mock import Mock
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import (
def test_LongitudeFormatter_number_format():
    formatter = LongitudeFormatter(number_format='.2f', dms=False, dateline_direction_label=True)
    p = ccrs.PlateCarree()
    formatter.set_axis(Mock(axes=Mock(GeoAxes, projection=p)))
    test_ticks = [-180, -120, -60, 0, 60, 120, 180]
    result = [formatter(tick) for tick in test_ticks]
    expected = ['180.00°W', '120.00°W', '60.00°W', '0.00°', '60.00°E', '120.00°E', '180.00°E']
    assert result == expected