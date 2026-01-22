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
def test_quadcontourset_reuse():
    x, y = np.meshgrid([0.0, 1.0], [0.0, 1.0])
    z = x + y
    fig, ax = plt.subplots()
    qcs1 = ax.contourf(x, y, z)
    qcs2 = ax.contour(x, y, z)
    assert qcs2._contour_generator != qcs1._contour_generator
    qcs3 = ax.contour(qcs1, z)
    assert qcs3._contour_generator == qcs1._contour_generator