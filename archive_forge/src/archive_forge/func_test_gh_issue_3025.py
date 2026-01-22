import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_gh_issue_3025():
    """Github issue #3025 - improper merging of labels"""
    d = np.zeros((60, 320))
    d[:, :257] = 1
    d[:, 260:] = 1
    d[36, 257] = 1
    d[35, 258] = 1
    d[35, 259] = 1
    assert ndimage.label(d, np.ones((3, 3)))[1] == 1