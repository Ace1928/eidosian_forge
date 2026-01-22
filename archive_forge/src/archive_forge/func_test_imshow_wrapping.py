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
def test_imshow_wrapping():
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0.0))
    ax.imshow(np.random.random((10, 10)), transform=ccrs.PlateCarree(), extent=(0, 360, -90, 90))
    assert ax.get_xlim() == (-180, 180)