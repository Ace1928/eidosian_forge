import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import cartopy.crs as ccrs
def test_limits_pcolor():
    xs, ys = np.meshgrid(np.linspace(250, 350, 15), np.linspace(-45, 45, 20))
    data = np.sin(xs * ys * 10000000.0)[:-1, :-1]
    resulting_extent = np.array([[250 - 180, -45.0], [-10.0 + 180, 45.0]])
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.pcolor(xs, ys, data, transform=ccrs.PlateCarree(180))
    assert_array_almost_equal(ax.dataLim, resulting_extent)
    plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.pcolormesh(xs, ys, data, transform=ccrs.PlateCarree(180))
    assert_array_almost_equal(ax.dataLim, resulting_extent)