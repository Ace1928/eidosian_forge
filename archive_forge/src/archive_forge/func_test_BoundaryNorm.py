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
def test_BoundaryNorm():
    """
    GitHub issue #1258: interpolation was failing with numpy
    1.7 pre-release.
    """
    boundaries = [0, 1.1, 2.2]
    vals = [-1, 0, 1, 2, 2.2, 4]
    expected = [-1, 0, 0, 1, 2, 2]
    ncolors = len(boundaries) - 1
    bn = mcolors.BoundaryNorm(boundaries, ncolors)
    assert_array_equal(bn(vals), expected)
    expected = [-1, 0, 0, 2, 3, 3]
    ncolors = len(boundaries)
    bn = mcolors.BoundaryNorm(boundaries, ncolors)
    assert_array_equal(bn(vals), expected)
    expected = [-1, 1, 1, 1, 3, 3]
    bn = mcolors.BoundaryNorm([0, 2.2], ncolors)
    assert_array_equal(bn(vals), expected)
    boundaries = [0, 1, 2, 3]
    vals = [-1, 0.1, 1.1, 2.2, 4]
    ncolors = 5
    expected = [-1, 0, 2, 4, 5]
    bn = mcolors.BoundaryNorm(boundaries, ncolors)
    assert_array_equal(bn(vals), expected)
    boundaries = [0, 1, 2]
    vals = [-1, 0.1, 1.1, 2.2]
    bn = mcolors.BoundaryNorm(boundaries, 2)
    expected = [-1, 0, 1, 2]
    for v, ex in zip(vals, expected):
        ret = bn(v)
        assert isinstance(ret, int)
        assert_array_equal(ret, ex)
        assert_array_equal(bn([v]), ex)
    bn = mcolors.BoundaryNorm(boundaries, 3)
    expected = [-1, 0, 2, 3]
    for v, ex in zip(vals, expected):
        ret = bn(v)
        assert isinstance(ret, int)
        assert_array_equal(ret, ex)
        assert_array_equal(bn([v]), ex)
    bn = mcolors.BoundaryNorm(boundaries, 3, clip=True)
    expected = [0, 0, 2, 2]
    for v, ex in zip(vals, expected):
        ret = bn(v)
        assert isinstance(ret, int)
        assert_array_equal(ret, ex)
        assert_array_equal(bn([v]), ex)
    boundaries = [0, 1.1, 2.2]
    vals = np.ma.masked_invalid([-1.0, np.nan, 0, 1.4, 9])
    ncolors = len(boundaries) - 1
    bn = mcolors.BoundaryNorm(boundaries, ncolors)
    expected = np.ma.masked_array([-1, -99, 0, 1, 2], mask=[0, 1, 0, 0, 0])
    assert_array_equal(bn(vals), expected)
    bn = mcolors.BoundaryNorm(boundaries, len(boundaries))
    expected = np.ma.masked_array([-1, -99, 0, 2, 3], mask=[0, 1, 0, 0, 0])
    assert_array_equal(bn(vals), expected)
    vals = np.ma.masked_invalid([np.inf, np.nan])
    assert np.all(bn(vals).mask)
    vals = np.ma.masked_invalid([np.inf])
    assert np.all(bn(vals).mask)
    with pytest.raises(ValueError, match='not compatible'):
        mcolors.BoundaryNorm(np.arange(4), 5, extend='both', clip=True)
    with pytest.raises(ValueError, match='ncolors must equal or exceed'):
        mcolors.BoundaryNorm(np.arange(4), 2)
    with pytest.raises(ValueError, match='ncolors must equal or exceed'):
        mcolors.BoundaryNorm(np.arange(4), 3, extend='min')
    with pytest.raises(ValueError, match='ncolors must equal or exceed'):
        mcolors.BoundaryNorm(np.arange(4), 4, extend='both')
    bounds = [1, 2, 3]
    cmap = mpl.colormaps['viridis']
    mynorm = mcolors.BoundaryNorm(bounds, cmap.N, extend='both')
    refnorm = mcolors.BoundaryNorm([0] + bounds + [4], cmap.N)
    x = np.random.randn(100) * 10 + 2
    ref = refnorm(x)
    ref[ref == 0] = -1
    ref[ref == cmap.N - 1] = cmap.N
    assert_array_equal(mynorm(x), ref)
    cmref = mcolors.ListedColormap(['blue', 'red'])
    cmref.set_over('black')
    cmref.set_under('white')
    cmshould = mcolors.ListedColormap(['white', 'blue', 'red', 'black'])
    assert mcolors.same_color(cmref.get_over(), 'black')
    assert mcolors.same_color(cmref.get_under(), 'white')
    refnorm = mcolors.BoundaryNorm(bounds, cmref.N)
    mynorm = mcolors.BoundaryNorm(bounds, cmshould.N, extend='both')
    assert mynorm.vmin == refnorm.vmin
    assert mynorm.vmax == refnorm.vmax
    assert mynorm(bounds[0] - 0.1) == -1
    assert mynorm(bounds[0] + 0.1) == 1
    assert mynorm(bounds[-1] - 0.1) == cmshould.N - 2
    assert mynorm(bounds[-1] + 0.1) == cmshould.N
    x = [-1, 1.2, 2.3, 9.6]
    assert_array_equal(cmshould(mynorm(x)), cmshould([0, 1, 2, 3]))
    x = np.random.randn(100) * 10 + 2
    assert_array_equal(cmshould(mynorm(x)), cmref(refnorm(x)))
    cmref = mcolors.ListedColormap(['blue', 'red'])
    cmref.set_under('white')
    cmshould = mcolors.ListedColormap(['white', 'blue', 'red'])
    assert mcolors.same_color(cmref.get_under(), 'white')
    assert cmref.N == 2
    assert cmshould.N == 3
    refnorm = mcolors.BoundaryNorm(bounds, cmref.N)
    mynorm = mcolors.BoundaryNorm(bounds, cmshould.N, extend='min')
    assert mynorm.vmin == refnorm.vmin
    assert mynorm.vmax == refnorm.vmax
    x = [-1, 1.2, 2.3]
    assert_array_equal(cmshould(mynorm(x)), cmshould([0, 1, 2]))
    x = np.random.randn(100) * 10 + 2
    assert_array_equal(cmshould(mynorm(x)), cmref(refnorm(x)))
    cmref = mcolors.ListedColormap(['blue', 'red'])
    cmref.set_over('black')
    cmshould = mcolors.ListedColormap(['blue', 'red', 'black'])
    assert mcolors.same_color(cmref.get_over(), 'black')
    assert cmref.N == 2
    assert cmshould.N == 3
    refnorm = mcolors.BoundaryNorm(bounds, cmref.N)
    mynorm = mcolors.BoundaryNorm(bounds, cmshould.N, extend='max')
    assert mynorm.vmin == refnorm.vmin
    assert mynorm.vmax == refnorm.vmax
    x = [1.2, 2.3, 4]
    assert_array_equal(cmshould(mynorm(x)), cmshould([0, 1, 2]))
    x = np.random.randn(100) * 10 + 2
    assert_array_equal(cmshould(mynorm(x)), cmref(refnorm(x)))