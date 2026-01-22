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
def test_google_tile_styles():
    """
    Tests that setting the Google Maps tile style works as expected.

    This is essentially just assures information is properly propagated through
    the class structure.
    """
    reference_url = 'https://mts0.google.com/vt/lyrs={style}@177000000&hl=en&src=api&x=1&y=2&z=3&s=G'
    tile = ['1', '2', '3']
    gt = cimgt.GoogleTiles()
    url = gt._image_url(tile)
    assert reference_url.format(style='m') == url
    gt = cimgt.GoogleTiles(style='street')
    url = gt._image_url(tile)
    assert reference_url.format(style='m') == url
    gt = cimgt.GoogleTiles(style='satellite')
    url = gt._image_url(tile)
    assert reference_url.format(style='s') == url
    gt = cimgt.GoogleTiles(style='terrain')
    url = gt._image_url(tile)
    assert reference_url.format(style='t') == url
    gt = cimgt.GoogleTiles(style='only_streets')
    url = gt._image_url(tile)
    assert reference_url.format(style='h') == url
    with pytest.raises(ValueError):
        cimgt.GoogleTiles(style='random_style')