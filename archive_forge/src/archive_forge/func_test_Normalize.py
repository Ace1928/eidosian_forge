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
def test_Normalize():
    norm = mcolors.Normalize()
    vals = np.arange(-10, 10, 1, dtype=float)
    _inverse_tester(norm, vals)
    _scalar_tester(norm, vals)
    _mask_tester(norm, vals)
    vals = np.array([-128, 127], dtype=np.int8)
    norm = mcolors.Normalize(vals.min(), vals.max())
    assert_array_equal(norm(vals), [0, 1])
    vals = np.array([1.2345678901, 9.8928432109], dtype=np.longdouble)
    norm = mcolors.Normalize(vals[0], vals[1])
    assert norm(vals).dtype == np.longdouble
    assert_array_equal(norm(vals), [0, 1])
    eps = np.finfo(np.longdouble).resolution
    norm = plt.Normalize(1, 1 + 100 * eps)
    assert_array_almost_equal(norm(1 + 50 * eps), 0.5, decimal=3)