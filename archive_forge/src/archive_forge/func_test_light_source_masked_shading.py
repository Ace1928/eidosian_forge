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
def test_light_source_masked_shading():
    """
    Array comparison test for a surface with a masked portion. Ensures that
    we don't wind up with "fringes" of odd colors around masked regions.
    """
    y, x = np.mgrid[-1.2:1.2:8j, -1.2:1.2:8j]
    z = 10 * np.cos(x ** 2 + y ** 2)
    z = np.ma.masked_greater(z, 9.9)
    cmap = plt.cm.copper
    ls = mcolors.LightSource(315, 45)
    rgb = ls.shade(z, cmap)
    expect = np.array([[[0.0, 0.46, 0.91, 0.91, 0.84, 0.64, 0.29, 0.0], [0.46, 0.96, 1.0, 1.0, 1.0, 0.97, 0.67, 0.18], [0.91, 1.0, 1.0, 1.0, 1.0, 1.0, 0.96, 0.36], [0.91, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.51], [0.84, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.44], [0.64, 0.97, 1.0, 1.0, 1.0, 1.0, 0.94, 0.09], [0.29, 0.67, 0.96, 1.0, 1.0, 0.94, 0.38, 0.01], [0.0, 0.18, 0.36, 0.51, 0.44, 0.09, 0.01, 0.0]], [[0.0, 0.29, 0.61, 0.75, 0.64, 0.41, 0.18, 0.0], [0.29, 0.81, 0.95, 0.93, 0.85, 0.68, 0.4, 0.11], [0.61, 0.95, 1.0, 0.78, 0.78, 0.77, 0.52, 0.22], [0.75, 0.93, 0.78, 0.0, 0.0, 0.78, 0.54, 0.19], [0.64, 0.85, 0.78, 0.0, 0.0, 0.78, 0.45, 0.08], [0.41, 0.68, 0.77, 0.78, 0.78, 0.55, 0.25, 0.02], [0.18, 0.4, 0.52, 0.54, 0.45, 0.25, 0.0, 0.0], [0.0, 0.11, 0.22, 0.19, 0.08, 0.02, 0.0, 0.0]], [[0.0, 0.19, 0.39, 0.48, 0.41, 0.26, 0.12, 0.0], [0.19, 0.52, 0.73, 0.78, 0.66, 0.46, 0.26, 0.07], [0.39, 0.73, 0.95, 0.5, 0.5, 0.53, 0.3, 0.14], [0.48, 0.78, 0.5, 0.0, 0.0, 0.5, 0.23, 0.12], [0.41, 0.66, 0.5, 0.0, 0.0, 0.5, 0.11, 0.05], [0.26, 0.46, 0.53, 0.5, 0.5, 0.11, 0.03, 0.01], [0.12, 0.26, 0.3, 0.23, 0.11, 0.03, 0.0, 0.0], [0.0, 0.07, 0.14, 0.12, 0.05, 0.01, 0.0, 0.0]], [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]]).T
    assert_array_almost_equal(rgb, expect, decimal=2)