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
def test_2d_to_rgba():
    color = np.array([0.1, 0.2, 0.3])
    rgba_1d = mcolors.to_rgba(color.reshape(-1))
    rgba_2d = mcolors.to_rgba(color.reshape((1, -1)))
    assert rgba_1d == rgba_2d