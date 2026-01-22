import os
import pathlib
import tempfile
import numpy as np
import pytest
from skimage import io
from skimage._shared.testing import assert_array_equal, fetch
from skimage.data import data_dir
def test_imread_pathlib_tiff():
    """Tests reading from Path object (issue gh-5545)."""
    fname = fetch('data/multipage.tif')
    expected = io.imread(fname)
    path = pathlib.Path(fname)
    img = io.imread(path)
    assert img.shape == (2, 15, 10)
    assert_array_equal(expected, img)