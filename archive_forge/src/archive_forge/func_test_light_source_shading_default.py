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
def test_light_source_shading_default():
    """
    Array comparison test for the default "hsv" blend mode. Ensure the
    default result doesn't change without warning.
    """
    y, x = np.mgrid[-1.2:1.2:8j, -1.2:1.2:8j]
    z = 10 * np.cos(x ** 2 + y ** 2)
    cmap = plt.cm.copper
    ls = mcolors.LightSource(315, 45)
    rgb = ls.shade(z, cmap)
    expect = np.array([[[0.0, 0.45, 0.9, 0.9, 0.82, 0.62, 0.28, 0.0], [0.45, 0.94, 0.99, 1.0, 1.0, 0.96, 0.65, 0.17], [0.9, 0.99, 1.0, 1.0, 1.0, 1.0, 0.94, 0.35], [0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.49], [0.82, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.41], [0.62, 0.96, 1.0, 1.0, 1.0, 1.0, 0.9, 0.07], [0.28, 0.65, 0.94, 1.0, 1.0, 0.9, 0.35, 0.01], [0.0, 0.17, 0.35, 0.49, 0.41, 0.07, 0.01, 0.0]], [[0.0, 0.28, 0.59, 0.72, 0.62, 0.4, 0.18, 0.0], [0.28, 0.78, 0.93, 0.92, 0.83, 0.66, 0.39, 0.11], [0.59, 0.93, 0.99, 1.0, 0.92, 0.75, 0.5, 0.21], [0.72, 0.92, 1.0, 0.99, 0.93, 0.76, 0.51, 0.18], [0.62, 0.83, 0.92, 0.93, 0.87, 0.68, 0.42, 0.08], [0.4, 0.66, 0.75, 0.76, 0.68, 0.52, 0.23, 0.02], [0.18, 0.39, 0.5, 0.51, 0.42, 0.23, 0.0, 0.0], [0.0, 0.11, 0.21, 0.18, 0.08, 0.02, 0.0, 0.0]], [[0.0, 0.18, 0.38, 0.46, 0.39, 0.26, 0.11, 0.0], [0.18, 0.5, 0.7, 0.75, 0.64, 0.44, 0.25, 0.07], [0.38, 0.7, 0.91, 0.98, 0.81, 0.51, 0.29, 0.13], [0.46, 0.75, 0.98, 0.96, 0.84, 0.48, 0.22, 0.12], [0.39, 0.64, 0.81, 0.84, 0.71, 0.31, 0.11, 0.05], [0.26, 0.44, 0.51, 0.48, 0.31, 0.1, 0.03, 0.01], [0.11, 0.25, 0.29, 0.22, 0.11, 0.03, 0.0, 0.0], [0.0, 0.07, 0.13, 0.12, 0.05, 0.01, 0.0, 0.0]], [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]]).T
    assert_array_almost_equal(rgb, expect, decimal=2)