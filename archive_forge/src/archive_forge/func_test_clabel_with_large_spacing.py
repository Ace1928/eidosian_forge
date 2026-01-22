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
def test_clabel_with_large_spacing():
    x = y = np.arange(-3.0, 3.01, 0.05)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-X ** 2 - Y ** 2)
    fig, ax = plt.subplots()
    contourset = ax.contour(X, Y, Z, levels=[0.01, 0.2, 0.5, 0.8])
    ax.clabel(contourset, inline_spacing=100)