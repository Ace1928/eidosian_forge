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
def test_colormap_invalid():
    """
    GitHub issue #9892: Handling of nan's were getting mapped to under
    rather than bad. This tests to make sure all invalid values
    (-inf, nan, inf) are mapped respectively to (under, bad, over).
    """
    cmap = mpl.colormaps['plasma']
    x = np.array([-np.inf, -1, 0, np.nan, 0.7, 2, np.inf])
    expected = np.array([[0.050383, 0.029803, 0.527975, 1.0], [0.050383, 0.029803, 0.527975, 1.0], [0.050383, 0.029803, 0.527975, 1.0], [0.0, 0.0, 0.0, 0.0], [0.949217, 0.517763, 0.295662, 1.0], [0.940015, 0.975158, 0.131326, 1.0], [0.940015, 0.975158, 0.131326, 1.0]])
    assert_array_equal(cmap(x), expected)
    expected = np.array([[0.0, 0.0, 0.0, 0.0], [0.050383, 0.029803, 0.527975, 1.0], [0.050383, 0.029803, 0.527975, 1.0], [0.0, 0.0, 0.0, 0.0], [0.949217, 0.517763, 0.295662, 1.0], [0.940015, 0.975158, 0.131326, 1.0], [0.0, 0.0, 0.0, 0.0]])
    assert_array_equal(cmap(np.ma.masked_invalid(x)), expected)
    assert_array_equal(cmap(-np.inf), cmap(0))
    assert_array_equal(cmap(np.inf), cmap(1.0))
    assert_array_equal(cmap(np.nan), [0.0, 0.0, 0.0, 0.0])