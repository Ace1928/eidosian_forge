import re
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.mpl import _MPL_38
from cartopy.tests.conftest import requires_scipy
@requires_scipy
@pytest.mark.natural_earth
@pytest.mark.mpl_image_compare(filename='barbs_regrid_with_extent.png', tolerance=0.54)
def test_barbs_regrid_with_extent():
    x = np.arange(-60, 42.5, 2.5)
    y = np.arange(30, 72.5, 2.5)
    x2d, y2d = np.meshgrid(x, y)
    u = 40 * np.cos(np.deg2rad(y2d))
    v = 40 * np.cos(2.0 * np.deg2rad(x2d))
    mag = (u ** 2 + v ** 2) ** 0.5
    plot_extent = [-60, 40, 30, 70]
    target_extent = [-3000000.0, 2000000.0, -6000000.0, -2500000.0]
    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(projection=ccrs.NorthPolarStereo())
    ax.set_extent(plot_extent, crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.barbs(x, y, u, v, mag, transform=ccrs.PlateCarree(), length=4, linewidth=0.25, regrid_shape=10, target_extent=target_extent)
    return fig