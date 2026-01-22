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
def test_FuncNorm():

    def forward(x):
        return x ** 2

    def inverse(x):
        return np.sqrt(x)
    norm = mcolors.FuncNorm((forward, inverse), vmin=0, vmax=10)
    expected = np.array([0, 0.25, 1])
    input = np.array([0, 5, 10])
    assert_array_almost_equal(norm(input), expected)
    assert_array_almost_equal(norm.inverse(expected), input)

    def forward(x):
        return np.log10(x)

    def inverse(x):
        return 10 ** x
    norm = mcolors.FuncNorm((forward, inverse), vmin=0.1, vmax=10)
    lognorm = mcolors.LogNorm(vmin=0.1, vmax=10)
    assert_array_almost_equal(norm([0.2, 5, 10]), lognorm([0.2, 5, 10]))
    assert_array_almost_equal(norm.inverse([0.2, 5, 10]), lognorm.inverse([0.2, 5, 10]))