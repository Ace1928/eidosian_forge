import types
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pytest
import shapely.geometry as sgeom
from cartopy import config
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from cartopy.tests.conftest import _HAS_PYKDTREE_OR_SCIPY
import cartopy.tests.test_img_tiles as ctest_tiles
def test_imshow_rgb():
    z = np.full((100, 100, 3), 0.5)
    plt_crs = ccrs.LambertAzimuthalEqualArea()
    latlon_crs = ccrs.PlateCarree()
    ax = plt.axes(projection=plt_crs)
    ax.set_extent([-30, -20, 60, 70], crs=latlon_crs)
    img = ax.imshow(z, extent=[-26, -24, 64, 66], transform=latlon_crs)
    assert sum(img.get_array().data[:, 0, 3]) == 0