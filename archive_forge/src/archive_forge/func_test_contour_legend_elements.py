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
def test_contour_legend_elements():
    x = np.arange(1, 10)
    y = x.reshape(-1, 1)
    h = x * y
    colors = ['blue', '#00FF00', 'red']
    cs = plt.contour(h, levels=[10, 30, 50], colors=colors, extend='both')
    artists, labels = cs.legend_elements()
    assert labels == ['$x = 10.0$', '$x = 30.0$', '$x = 50.0$']
    assert all((isinstance(a, mpl.lines.Line2D) for a in artists))
    assert all((same_color(a.get_color(), c) for a, c in zip(artists, colors)))