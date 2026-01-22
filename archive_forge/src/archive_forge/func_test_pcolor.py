from unittest import mock
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes, GeoAxesSubplot, InterProjectionTransform
def test_pcolor(self):
    with pytest.raises(ValueError):
        self.ax.pcolor(self.data, transform=ccrs.Geodetic())