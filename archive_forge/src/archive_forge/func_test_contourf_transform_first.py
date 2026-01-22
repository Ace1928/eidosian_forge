import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
import cartopy.crs as ccrs
from cartopy.tests.conftest import requires_scipy
@pytest.mark.parametrize('func', ['contour', 'contourf'])
def test_contourf_transform_first(func):
    """Test the fast-path option for filled contours."""
    x = np.arange(360)
    y = np.arange(-25, 26)
    xx, yy = np.meshgrid(x, y)
    z = xx + yy ** 2
    ax = plt.axes(projection=ccrs.PlateCarree())
    test_func = getattr(ax, func)
    with pytest.raises(ValueError, match='The X and Y arguments must be provided'):
        test_func(z, transform=ccrs.PlateCarree(), transform_first=True)
    with pytest.raises(ValueError, match='The X and Y arguments must be gridded'):
        test_func(x, y, z, transform=ccrs.PlateCarree(), transform_first=True)
    test_func(xx, yy, z, transform=ccrs.PlateCarree(), transform_first=True)
    assert_array_almost_equal(ax.get_extent(), (-179, 180, -25, 25))
    test_func(xx, yy, z, transform=ccrs.PlateCarree(), transform_first=False)
    assert_array_almost_equal(ax.get_extent(), (-180, 180, -25, 25))