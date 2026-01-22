import io
from unittest import mock
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
def test_pcolormesh_arg_interpolation():
    x = [359, 1, 3]
    y = [-10, 10]
    xs, ys = np.meshgrid(x, y)
    z = np.zeros(xs.shape)
    ax = plt.subplot(2, 1, 1, projection=ccrs.PlateCarree())
    coll = ax.pcolormesh(xs, ys, z, shading='auto', transform=ccrs.PlateCarree())
    expected = np.array([[[358, -20], [360, -20], [2, -20], [4, -20]], [[358, 0], [360, 0], [2, 0], [4, 0]], [[358, 20], [360, 20], [2, 20], [4, 20]]])
    np.testing.assert_array_almost_equal(expected, coll._coordinates)