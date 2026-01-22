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
@pytest.mark.skipif(geos_version == (3, 9, 0), reason='GEOS intersection bug')
@pytest.mark.natural_earth
@pytest.mark.mpl_image_compare(filename='gridliner_labels_tight.png', tolerance=2.9)
def test_grid_labels_tight():
    fig = plt.figure(figsize=(7, 5))
    crs_pc = ccrs.PlateCarree()
    crs_merc = ccrs.Mercator()
    ax = fig.add_subplot(2, 2, 1, projection=crs_pc)
    ax.coastlines(resolution='110m')
    ax.gridlines(draw_labels=True)
    ax = fig.add_subplot(2, 2, 2, projection=crs_merc)
    ax.coastlines(resolution='110m')
    ax.gridlines(draw_labels=True)
    ax = fig.add_subplot(2, 2, 3, projection=crs_pc)
    ax.set_extent([-20, 10.0, 45.0, 70.0])
    ax.coastlines(resolution='110m')
    ax.gridlines(draw_labels=True)
    ax = fig.add_subplot(2, 2, 4, projection=crs_merc)
    ax.set_extent([-20, 10.0, 45.0, 70.0], crs=crs_pc)
    ax.coastlines(resolution='110m')
    gl = ax.gridlines(draw_labels=True)
    gl.rotate_labels = False
    fig.tight_layout()
    num_gridliners_drawn = 0
    for ax in fig.axes:
        for artist in ax.artists:
            if isinstance(artist, Gridliner) and getattr(artist, '_drawn', False):
                num_gridliners_drawn += 1
    assert num_gridliners_drawn == 4
    return fig