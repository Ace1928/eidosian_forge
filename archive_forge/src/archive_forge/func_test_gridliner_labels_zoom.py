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
def test_gridliner_labels_zoom():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_global()
    gl = ax.gridlines(draw_labels=True)
    fig.draw_without_rendering()
    labels = [a.get_text() for a in gl.bottom_label_artists if a.get_visible()]
    assert labels == ['180°', '120°W', '60°W', '0°', '60°E', '120°E', '180°']
    assert len(gl._all_labels) == 24
    assert gl._labels == gl._all_labels
    ax.set_extent([-20, 10.0, 45.0, 70.0])
    fig.draw_without_rendering()
    labels = [a.get_text() for a in gl.bottom_label_artists if a.get_visible()]
    assert labels == ['15°W', '10°W', '5°W', '0°', '5°E']
    assert len(gl._all_labels) == 24
    assert gl._labels == gl._all_labels[:20]