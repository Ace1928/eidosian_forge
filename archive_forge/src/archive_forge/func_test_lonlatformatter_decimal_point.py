from unittest.mock import Mock
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import (
def test_lonlatformatter_decimal_point():
    xticker = LongitudeFormatter(decimal_point=',', number_format='0.2f')
    yticker = LatitudeFormatter(decimal_point=',', number_format='0.2f')
    assert xticker(-10) == '10,00°W'
    assert yticker(-10) == '10,00°S'