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
def test_azuremaps_get_image():
    try:
        api_key = os.environ['AZURE_MAPS_SUBSCRIPTION_KEY']
    except KeyError:
        pytest.skip('AZURE_MAPS_SUBSCRIPTION_KEY environment variable is unset.')
    am1 = cimgt.AzureMapsTiles(api_key, tileset_id='microsoft.imagery')
    am2 = cimgt.AzureMapsTiles(api_key, tileset_id='microsoft.base.road')
    tile = (500, 300, 10)
    img1, extent1, _ = am1.get_image(tile)
    img2, extent2, _ = am2.get_image(tile)
    assert img1 != img2
    assert extent1 == extent2