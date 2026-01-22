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
def test_resampled():
    """
    GitHub issue #6025 pointed to incorrect ListedColormap.resampled;
    here we test the method for LinearSegmentedColormap as well.
    """
    n = 101
    colorlist = np.empty((n, 4), float)
    colorlist[:, 0] = np.linspace(0, 1, n)
    colorlist[:, 1] = 0.2
    colorlist[:, 2] = np.linspace(1, 0, n)
    colorlist[:, 3] = 0.7
    lsc = mcolors.LinearSegmentedColormap.from_list('lsc', colorlist)
    lc = mcolors.ListedColormap(colorlist)
    for cmap in [lsc, lc]:
        cmap.set_under('r')
        cmap.set_over('g')
        cmap.set_bad('b')
    lsc3 = lsc.resampled(3)
    lc3 = lc.resampled(3)
    expected = np.array([[0.0, 0.2, 1.0, 0.7], [0.5, 0.2, 0.5, 0.7], [1.0, 0.2, 0.0, 0.7]], float)
    assert_array_almost_equal(lsc3([0, 0.5, 1]), expected)
    assert_array_almost_equal(lc3([0, 0.5, 1]), expected)
    assert_array_almost_equal(lsc(np.inf), lsc3(np.inf))
    assert_array_almost_equal(lsc(-np.inf), lsc3(-np.inf))
    assert_array_almost_equal(lsc(np.nan), lsc3(np.nan))
    assert_array_almost_equal(lc(np.inf), lc3(np.inf))
    assert_array_almost_equal(lc(-np.inf), lc3(-np.inf))
    assert_array_almost_equal(lc(np.nan), lc3(np.nan))