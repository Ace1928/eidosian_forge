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
def test_contourf_symmetric_locator():
    z = np.arange(12).reshape((3, 4))
    locator = plt.MaxNLocator(nbins=4, symmetric=True)
    cs = plt.contourf(z, locator=locator)
    assert_array_almost_equal(cs.levels, np.linspace(-12, 12, 5))