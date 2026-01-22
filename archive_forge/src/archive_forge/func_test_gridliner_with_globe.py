import io
from unittest import mock
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pytest
from shapely.geos import geos_version
import cartopy.crs as ccrs
from cartopy.mpl import _MPL_36
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.gridliner import (
from cartopy.mpl.ticker import LongitudeFormatter, LongitudeLocator
def test_gridliner_with_globe():
    fig = plt.figure()
    proj = ccrs.PlateCarree(globe=ccrs.Globe(semimajor_axis=12345))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    gl = ax.gridlines()
    fig.draw_without_rendering()
    assert gl in ax.artists