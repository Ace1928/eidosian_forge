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
def test_img_pickle_round_trip():
    """Check that __getstate__ for Img instances is working correctly."""
    img = cimg_nest.Img('imaginary file', (0, 1, 2, 3), 'lower', (1, 2))
    img_from_pickle = pickle.loads(pickle.dumps(img))
    assert img == img_from_pickle
    assert hasattr(img_from_pickle, '_bbox')