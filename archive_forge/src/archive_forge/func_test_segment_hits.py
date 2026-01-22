import itertools
import platform
import timeit
from types import SimpleNamespace
from cycler import cycler
import numpy as np
from numpy.testing import assert_array_equal
import pytest
import matplotlib
import matplotlib as mpl
from matplotlib import _path
import matplotlib.lines as mlines
from matplotlib.markers import MarkerStyle
from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_segment_hits():
    """Test a problematic case."""
    cx, cy = (553, 902)
    x, y = (np.array([553.0, 553.0]), np.array([95.0, 947.0]))
    radius = 6.94
    assert_array_equal(mlines.segment_hits(cx, cy, x, y, radius), [0])