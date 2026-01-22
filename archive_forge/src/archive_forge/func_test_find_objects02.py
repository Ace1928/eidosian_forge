import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_find_objects02():
    data = np.zeros([], dtype=int)
    out = ndimage.find_objects(data)
    assert_(out == [])