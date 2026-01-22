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
def test_ordnance_survey_tile_styles():
    """
    Tests that setting the Ordnance Survey tile style works as expected.

    This is essentially just assures information is properly propagated through
    the class structure.
    """
    dummy_apikey = 'None'
    ref_url = 'https://api.os.uk/maps/raster/v1/zxy/{layer}/{z}/{x}/{y}.png?key=None'
    tile = ['1', '2', '3']
    ordsurvey = cimgt.OrdnanceSurvey(dummy_apikey)
    url = ordsurvey._image_url(tile)
    assert url == ref_url.format(layer='Road_3857', z=tile[2], y=tile[1], x=tile[0])
    for layer in ('Road_3857', 'Light_3857', 'Outdoor_3857', 'Road', 'Light', 'Outdoor'):
        ordsurvey = cimgt.OrdnanceSurvey(dummy_apikey, layer=layer)
        url = ordsurvey._image_url(tile)
        layer = layer if layer.endswith('_3857') else layer + '_3857'
        assert url == ref_url.format(layer=layer, z=tile[2], y=tile[1], x=tile[0])
    with pytest.raises(ValueError):
        cimgt.OrdnanceSurvey(dummy_apikey, layer='random_style')