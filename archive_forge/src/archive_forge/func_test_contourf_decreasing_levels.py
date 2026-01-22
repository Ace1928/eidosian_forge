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
def test_contourf_decreasing_levels():
    z = [[0.1, 0.3], [0.5, 0.7]]
    plt.figure()
    with pytest.raises(ValueError):
        plt.contourf(z, [1.0, 0.0])