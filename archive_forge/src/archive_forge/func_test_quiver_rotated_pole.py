import re
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.mpl import _MPL_38
from cartopy.tests.conftest import requires_scipy
@pytest.mark.natural_earth
@pytest.mark.mpl_image_compare(filename='quiver_rotated_pole.png')
def test_quiver_rotated_pole():
    nx, ny = (22, 36)
    x = np.linspace(311.91998291, 391.11999512, nx, endpoint=True)
    y = np.linspace(-23.59000015, 24.81000137, ny, endpoint=True)
    x2d, y2d = np.meshgrid(x, y)
    u = np.cos(np.deg2rad(y2d))
    v = -2.0 * np.cos(2.0 * np.deg2rad(y2d)) * np.sin(np.deg2rad(x2d))
    mag = (u ** 2 + v ** 2) ** 0.5
    rp = ccrs.RotatedPole(pole_longitude=177.5, pole_latitude=37.5)
    plot_extent = [x[0], x[-1], y[0], y[-1]]
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(2, 1, 1, projection=rp)
    ax.set_extent(plot_extent, crs=rp)
    ax.coastlines()
    ax.quiver(x, y, u, v, mag)
    ax = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree())
    ax.set_extent(plot_extent, crs=rp)
    ax.coastlines()
    ax.quiver(x, y, u, v, mag, transform=rp)
    return fig