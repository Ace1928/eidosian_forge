import io
from unittest import mock
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
def test_pcolormesh_datalim():
    x = [359, 1, 3]
    y = [-10, 10]
    xs, ys = np.meshgrid(x, y)
    z = np.zeros(xs.shape)
    ax = plt.subplot(3, 1, 1, projection=ccrs.PlateCarree())
    coll = ax.pcolormesh(xs, ys, z, shading='auto', transform=ccrs.PlateCarree())
    coll_bbox = coll.get_datalim(ax.transData)
    np.testing.assert_array_equal(coll_bbox, [[-2, -20], [4, 20]])
    x = [-80, 0, 80]
    y = [-10, 10]
    xs, ys = np.meshgrid(x, y)
    ax = plt.subplot(3, 1, 2, projection=ccrs.PlateCarree())
    coll = ax.pcolormesh(xs, ys, z, shading='auto', transform=ccrs.PlateCarree())
    coll_bbox = coll.get_datalim(ax.transData)
    np.testing.assert_array_equal(coll_bbox, [[-120, -20], [120, 20]])
    x = [-10, 0, 10]
    y = [-10, 10]
    xs, ys = np.meshgrid(x, y)
    ax = plt.subplot(3, 1, 3, projection=ccrs.Orthographic())
    coll = ax.pcolormesh(xs, ys, z, shading='auto', transform=ccrs.PlateCarree())
    coll_bbox = coll.get_datalim(ax.transData)
    expected = [[-1650783.327873, -2181451.330891], [1650783.327873, 2181451.330891]]
    np.testing.assert_array_almost_equal(coll_bbox, expected)