import io
from pathlib import Path
import pickle
import shutil
import sys
import warnings
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from PIL import Image
import pytest
import shapely.geometry as sgeom
from cartopy import config
import cartopy.io.img_nest as cimg_nest
import cartopy.io.img_tiles as cimgt
@pytest.fixture
def nest_from_config(wmts_data):
    from_config = cimg_nest.NestedImageCollection.from_configuration
    files = [['aerial z0 test', _TEST_DATA_DIR / 'z_0'], ['aerial z1 test', _TEST_DATA_DIR / 'z_1']]
    crs = cimgt.GoogleTiles().crs
    nest_z0_z1 = from_config('aerial test', crs, files, glob_pattern='*.png', img_class=RoundedImg)
    return nest_z0_z1