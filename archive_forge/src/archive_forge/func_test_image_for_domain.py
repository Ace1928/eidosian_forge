import hashlib
import os
import types
import warnings
import numpy as np
from numpy.testing import assert_array_almost_equal as assert_arr_almost
import pytest
import shapely.geometry as sgeom
from cartopy import config
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
@pytest.mark.network
def test_image_for_domain():
    gt = cimgt.GoogleTiles()
    gt._image_url = types.MethodType(GOOGLE_IMAGE_URL_REPLACEMENT, gt)
    ll_target_domain = sgeom.box(-10, 50, 10, 60)
    multi_poly = gt.crs.project_geometry(ll_target_domain, ccrs.PlateCarree())
    target_domain = multi_poly.geoms[0]
    _, extent, _ = gt.image_for_domain(target_domain, 6)
    ll_extent = ccrs.Geodetic().transform_points(gt.crs, np.array(extent[:2]), np.array(extent[2:]))
    assert_arr_almost(ll_extent[:, :2], [[-11.25, 48.92249926], [11.25, 61.60639637]])