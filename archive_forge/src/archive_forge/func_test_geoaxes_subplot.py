from unittest import mock
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes, GeoAxesSubplot, InterProjectionTransform
def test_geoaxes_subplot():
    ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
    assert isinstance(ax, GeoAxesSubplot)