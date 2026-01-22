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
def test_allsegs_allkinds():
    x, y = np.meshgrid(np.arange(0, 10, 2), np.arange(0, 10, 2))
    z = np.sin(x) * np.cos(y)
    cs = plt.contour(x, y, z, levels=[0, 0.5])
    for result in [cs.allsegs, cs.allkinds]:
        assert len(result) == 2
        assert len(result[0]) == 5
        assert len(result[1]) == 4