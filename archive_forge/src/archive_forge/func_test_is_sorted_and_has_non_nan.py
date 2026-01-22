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
def test_is_sorted_and_has_non_nan():
    assert _path.is_sorted_and_has_non_nan(np.array([1, 2, 3]))
    assert _path.is_sorted_and_has_non_nan(np.array([1, np.nan, 3]))
    assert not _path.is_sorted_and_has_non_nan([3, 5] + [np.nan] * 100 + [0, 2])
    assert not _path.is_sorted_and_has_non_nan(np.array([33554432, 65536], '>i4'))
    n = 2 * mlines.Line2D._subslice_optim_min_size
    plt.plot([np.nan] * n, range(n))