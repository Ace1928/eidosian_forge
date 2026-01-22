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
@image_comparison(['levels_and_colors.png'])
def test_cmap_and_norm_from_levels_and_colors():
    plt.rcParams['pcolormesh.snap'] = False
    data = np.linspace(-2, 4, 49).reshape(7, 7)
    levels = [-1, 2, 2.5, 3]
    colors = ['red', 'green', 'blue', 'yellow', 'black']
    extend = 'both'
    cmap, norm = mcolors.from_levels_and_colors(levels, colors, extend=extend)
    ax = plt.axes()
    m = plt.pcolormesh(data, cmap=cmap, norm=norm)
    plt.colorbar(m)
    ax.tick_params(labelleft=False, labelbottom=False)