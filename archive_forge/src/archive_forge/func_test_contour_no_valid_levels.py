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
def test_contour_no_valid_levels():
    fig, ax = plt.subplots()
    ax.contour(np.random.rand(9, 9), levels=[])
    cs = ax.contour(np.arange(81).reshape((9, 9)), levels=[100])
    ax.clabel(cs, fmt={100: '%1.2f'})
    ax.contour(np.ones((9, 9)))