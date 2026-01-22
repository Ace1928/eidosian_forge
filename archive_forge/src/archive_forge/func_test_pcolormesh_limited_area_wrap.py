import re
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.mpl import _MPL_38
from cartopy.tests.conftest import requires_scipy
@pytest.mark.natural_earth
@pytest.mark.mpl_image_compare(filename='pcolormesh_limited_area_wrap.png', tolerance=1.83)
def test_pcolormesh_limited_area_wrap():
    nx, ny = (22, 36)
    xbnds = np.linspace(311.91998291, 391.11999512, nx, endpoint=True)
    ybnds = np.linspace(-23.59000015, 24.81000137, ny, endpoint=True)
    x, y = np.meshgrid(xbnds, ybnds)
    data = np.sin(np.deg2rad(x)) / 10.0 + np.exp(np.cos(np.deg2rad(y)))
    data = data[:-1, :-1]
    rp = ccrs.RotatedPole(pole_longitude=177.5, pole_latitude=37.5)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(2, 2, 1, projection=ccrs.PlateCarree())
    ax.pcolormesh(xbnds, ybnds, data, transform=rp, cmap='Spectral', snap=False)
    ax.coastlines()
    ax = fig.add_subplot(2, 2, 2, projection=ccrs.PlateCarree(180))
    ax.pcolormesh(xbnds, ybnds, data, transform=rp, cmap='Spectral', snap=False)
    ax.coastlines()
    ax.set_global()
    ax = fig.add_subplot(2, 2, 3, projection=ccrs.PlateCarree())
    ax.pcolormesh(x, y, data, transform=rp, cmap='Spectral', snap=False)
    ax.coastlines()
    ax.set_extent([-70, 0, 0, 80])
    ax = fig.add_subplot(2, 2, 4, projection=rp)
    ax.pcolormesh(xbnds, ybnds, data, transform=rp, cmap='Spectral', snap=False)
    ax.coastlines()
    return fig