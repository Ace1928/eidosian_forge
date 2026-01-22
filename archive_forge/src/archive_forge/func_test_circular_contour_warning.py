import datetime
import platform
import re
from unittest import mock
import contourpy
import numpy as np
from numpy.testing import (
import matplotlib as mpl
from matplotlib import pyplot as plt, rc_context, ticker
from matplotlib.colors import LogNorm, same_color
import matplotlib.patches as mpatches
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import pytest
def test_circular_contour_warning():
    x, y = np.meshgrid(np.linspace(-2, 2, 4), np.linspace(-2, 2, 4))
    r = np.hypot(x, y)
    plt.figure()
    cs = plt.contour(x, y, r)
    plt.clabel(cs)