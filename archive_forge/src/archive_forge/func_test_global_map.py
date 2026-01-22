import matplotlib.pyplot as plt
import pytest
import cartopy.crs as ccrs
from cartopy.mpl import _MPL_38
from cartopy.tests.conftest import _HAS_PYKDTREE_OR_SCIPY
@pytest.mark.natural_earth
@pytest.mark.mpl_image_compare(filename='global_map.png')
def test_global_map():
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()
    ax.stock_img()
    ax.coastlines()
    ax.plot(-0.08, 51.53, 'o', transform=ccrs.PlateCarree())
    ax.plot([-0.08, 132], [51.53, 43.17], transform=ccrs.PlateCarree())
    ax.plot([-0.08, 132], [51.53, 43.17], transform=ccrs.Geodetic())
    return fig