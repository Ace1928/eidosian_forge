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
@pytest.mark.parametrize('name', sorted(mpl.colormaps()))
def test_colormap_reversing(name):
    """
    Check the generated _lut data of a colormap and corresponding reversed
    colormap if they are almost the same.
    """
    cmap = mpl.colormaps[name]
    cmap_r = cmap.reversed()
    if not cmap_r._isinit:
        cmap._init()
        cmap_r._init()
    assert_array_almost_equal(cmap._lut[:-3], cmap_r._lut[-4::-1])
    assert_array_almost_equal(cmap(-np.inf), cmap_r(np.inf))
    assert_array_almost_equal(cmap(np.inf), cmap_r(-np.inf))
    assert_array_almost_equal(cmap(np.nan), cmap_r(np.nan))