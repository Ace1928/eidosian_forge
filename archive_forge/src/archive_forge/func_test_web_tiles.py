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
@pytest.mark.natural_earth
@pytest.mark.network
@pytest.mark.mpl_image_compare(filename='web_tiles.png', tolerance=5.91)
def test_web_tiles():
    extent = [-15, 0.1, 50, 60]
    target_domain = sgeom.Polygon([[extent[0], extent[1]], [extent[2], extent[1]], [extent[2], extent[3]], [extent[0], extent[3]], [extent[0], extent[1]]])
    map_prj = cimgt.GoogleTiles().crs
    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1, projection=map_prj)
    gt = cimgt.GoogleTiles()
    gt._image_url = types.MethodType(ctest_tiles.GOOGLE_IMAGE_URL_REPLACEMENT, gt)
    img, extent, origin = gt.image_for_domain(target_domain, 1)
    ax.imshow(np.array(img), extent=extent, transform=gt.crs, interpolation='bilinear', origin=origin)
    ax.coastlines(color='white')
    ax = fig.add_subplot(2, 2, 2, projection=map_prj)
    qt = cimgt.QuadtreeTiles()
    img, extent, origin = qt.image_for_domain(target_domain, 1)
    ax.imshow(np.array(img), extent=extent, transform=qt.crs, interpolation='bilinear', origin=origin)
    ax.coastlines(color='white')
    ax = fig.add_subplot(2, 2, 3, projection=map_prj)
    osm = cimgt.OSM()
    img, extent, origin = osm.image_for_domain(target_domain, 1)
    ax.imshow(np.array(img), extent=extent, transform=osm.crs, interpolation='bilinear', origin=origin)
    ax.coastlines()
    return fig