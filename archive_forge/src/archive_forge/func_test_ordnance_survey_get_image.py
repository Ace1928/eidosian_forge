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
def test_ordnance_survey_get_image():
    try:
        api_key = os.environ['ORDNANCE_SURVEY_API_KEY']
    except KeyError:
        pytest.skip('ORDNANCE_SURVEY_API_KEY environment variable is unset.')
    os1 = cimgt.OrdnanceSurvey(api_key, layer='Outdoor_3857')
    os2 = cimgt.OrdnanceSurvey(api_key, layer='Light_3857')
    tile = (500, 300, 10)
    img1, extent1, _ = os1.get_image(tile)
    img2, extent2, _ = os2.get_image(tile)
    assert img1 != img2
    assert extent1 == extent2