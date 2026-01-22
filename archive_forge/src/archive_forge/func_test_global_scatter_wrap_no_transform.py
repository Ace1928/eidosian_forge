import re
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.mpl import _MPL_38
from cartopy.tests.conftest import requires_scipy
@pytest.mark.natural_earth
@pytest.mark.mpl_image_compare(filename='global_scatter_wrap.png')
def test_global_scatter_wrap_no_transform():
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(zorder=0)
    x, y = np.meshgrid(np.linspace(0, 360), np.linspace(-90, 90))
    data = np.sin(np.sqrt(x ** 2 + y ** 2))
    ax.scatter(x, y, c=data)
    return ax.figure