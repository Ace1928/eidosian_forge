from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
from matplotlib.path import Path
import matplotlib.pyplot as plt
import numpy as np
import pytest
import shapely.geometry as sgeom
import cartopy.crs as ccrs
import cartopy.mpl.patch as cpatch
@pytest.mark.natural_earth
@pytest.mark.mpl_image_compare(filename='contour_with_interiors.png')
def test_contour_interiors():
    nx, ny = (10, 10)
    numlev = 2
    lons, lats = np.meshgrid(np.linspace(-50, 50, nx), np.linspace(-45, 45, ny))
    data = np.sin(np.sqrt(lons ** 2 + lats ** 2))
    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1, projection=ccrs.PlateCarree())
    ax.set_global()
    ax.contourf(lons, lats, data, numlev, transform=ccrs.PlateCarree())
    ax.coastlines()
    ax = fig.add_subplot(2, 2, 2, projection=ccrs.Robinson())
    ax.set_global()
    ax.contourf(lons, lats, data, numlev, transform=ccrs.PlateCarree())
    ax.coastlines()
    numlev = 2
    x, y = np.meshgrid(np.arange(-5.5, 5.5, 0.25), np.arange(-5.5, 5.5, 0.25))
    dim = x.shape[0]
    data = np.sin(np.sqrt(x ** 2 + y ** 2))
    lats = np.arange(dim) + 30
    lons = np.arange(dim) - 20
    ax = fig.add_subplot(2, 2, 3, projection=ccrs.PlateCarree())
    ax.set_global()
    ax.contourf(lons, lats, data, numlev, transform=ccrs.PlateCarree())
    ax.coastlines()
    ax = fig.add_subplot(2, 2, 4, projection=ccrs.Robinson())
    ax.set_global()
    ax.contourf(lons, lats, data, numlev, transform=ccrs.PlateCarree())
    ax.coastlines()
    return fig