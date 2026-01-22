import re
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.mpl import _MPL_38
from cartopy.tests.conftest import requires_scipy
@pytest.mark.natural_earth
@pytest.mark.mpl_image_compare(filename='global_hexbin_wrap.png', tolerance=0.5)
def test_global_hexbin_wrap_transform():
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(zorder=2)
    x, y = np.meshgrid(np.arange(0, 360), np.arange(-90, 91))
    x_wrap = np.where(x >= 180, x - 360, x)
    data = np.sin(np.sqrt(x_wrap ** 2 + y ** 2))
    ax.hexbin(x.flatten(), y.flatten(), C=data.flatten(), gridsize=20, zorder=1)
    return ax.figure