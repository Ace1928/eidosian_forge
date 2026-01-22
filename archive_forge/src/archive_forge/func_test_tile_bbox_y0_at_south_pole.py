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
def test_tile_bbox_y0_at_south_pole():
    tms = cimgt.MapQuestOpenAerial()
    assert_arr_almost(tms.tile_bbox(8, 6, 4, y0_at_north_pole=False), np.array(KNOWN_EXTENTS[8, 9, 4]).reshape([2, 2]))