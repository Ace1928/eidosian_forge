import copy
import itertools
import unittest.mock
from packaging.version import parse as parse_version
from io import BytesIO
import numpy as np
from PIL import Image
import pytest
import base64
from numpy.testing import assert_array_equal, assert_array_almost_equal
from matplotlib import cbook, cm
import matplotlib
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
import matplotlib.pyplot as plt
import matplotlib.scale as mscale
from matplotlib.rcsetup import cycler
from matplotlib.testing.decorators import image_comparison, check_figures_equal
@pytest.mark.parametrize('bytes', (True, False))
def test_scalarmappable_to_rgba(bytes):
    sm = cm.ScalarMappable()
    alpha_1 = 255 if bytes else 1
    x = np.ones((2, 3, 4), dtype=np.uint8)
    expected = x.copy() if bytes else x.astype(np.float32) / 255
    np.testing.assert_almost_equal(sm.to_rgba(x, bytes=bytes), expected)
    expected[..., 3] = alpha_1
    np.testing.assert_almost_equal(sm.to_rgba(x[..., :3], bytes=bytes), expected)
    xm = np.ma.masked_array(x, mask=np.zeros_like(x))
    xm.mask[0, 0, 0] = True
    expected = x.copy() if bytes else x.astype(np.float32) / 255
    expected[0, 0, 3] = 0
    np.testing.assert_almost_equal(sm.to_rgba(xm, bytes=bytes), expected)
    expected[..., 3] = alpha_1
    expected[0, 0, 3] = 0
    np.testing.assert_almost_equal(sm.to_rgba(xm[..., :3], bytes=bytes), expected)
    x = np.ones((2, 3, 4), dtype=float) * 0.5
    expected = (x * 255).astype(np.uint8) if bytes else x.copy()
    np.testing.assert_almost_equal(sm.to_rgba(x, bytes=bytes), expected)
    expected[..., 3] = alpha_1
    np.testing.assert_almost_equal(sm.to_rgba(x[..., :3], bytes=bytes), expected)
    xm = np.ma.masked_array(x, mask=np.zeros_like(x))
    xm.mask[0, 0, 0] = True
    expected = (x * 255).astype(np.uint8) if bytes else x.copy()
    expected[0, 0, 3] = 0
    np.testing.assert_almost_equal(sm.to_rgba(xm, bytes=bytes), expected)
    expected[..., 3] = alpha_1
    expected[0, 0, 3] = 0
    np.testing.assert_almost_equal(sm.to_rgba(xm[..., :3], bytes=bytes), expected)