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
@pytest.mark.parametrize('style,extension,resolution', [('alidade_smooth', 'png', ''), ('alidade_smooth', 'png', '@2x'), ('stamen_watercolor', 'jpg', '')])
def test_stadia_maps_tiles_api_url(style, extension, resolution):
    apikey = 'foo'
    tile = [0, 1, 2]
    exp_url = f'http://tiles.stadiamaps.com/tiles/{style}/2/0/1{resolution}.{extension}?api_key=foo'
    sample = cimgt.StadiaMapsTiles(apikey, style=style, resolution=resolution)
    url_str = sample._image_url(tile)
    assert url_str == exp_url