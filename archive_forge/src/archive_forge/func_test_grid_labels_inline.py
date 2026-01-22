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
@pytest.mark.parametrize('proj', TEST_PROJS)
@pytest.mark.mpl_image_compare(style='mpl20')
def test_grid_labels_inline(proj):
    fig = plt.figure()
    if isinstance(proj, tuple):
        proj, kwargs = proj
    else:
        kwargs = {}
    ax = fig.add_subplot(projection=proj(**kwargs))
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, auto_inline=True)
    ax.coastlines(resolution='110m')
    return fig