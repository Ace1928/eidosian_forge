import re
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.mpl import _MPL_38
from cartopy.tests.conftest import requires_scipy
@pytest.mark.natural_earth
@pytest.mark.mpl_image_compare(filename='quiver_plate_carree.png')
def test_quiver_plate_carree():
    x = np.arange(-60, 42.5, 2.5)
    y = np.arange(30, 72.5, 2.5)
    x2d, y2d = np.meshgrid(x, y)
    u = np.cos(np.deg2rad(y2d))
    v = np.cos(2.0 * np.deg2rad(x2d))
    mag = (u ** 2 + v ** 2) ** 0.5
    plot_extent = [-60, 40, 30, 70]
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(plot_extent, crs=ccrs.PlateCarree())
    ax.coastlines(resolution='110m')
    ax.quiver(x, y, u, v, mag)
    ax = fig.add_subplot(2, 1, 2, projection=ccrs.NorthPolarStereo())
    ax.set_extent(plot_extent, crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.quiver(x, y, u, v, mag, transform=ccrs.PlateCarree())
    return fig