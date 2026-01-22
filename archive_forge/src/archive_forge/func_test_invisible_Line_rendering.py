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
@pytest.mark.flaky(reruns=3)
def test_invisible_Line_rendering():
    """
    GitHub issue #1256 identified a bug in Line.draw method

    Despite visibility attribute set to False, the draw method was not
    returning early enough and some pre-rendering code was executed
    though not necessary.

    Consequence was an excessive draw time for invisible Line instances
    holding a large number of points (Npts> 10**6)
    """
    N = 10 ** 7
    x = np.linspace(0, 1, N)
    y = np.random.normal(size=N)
    fig = plt.figure()
    ax = plt.subplot()
    l = mlines.Line2D(x, y)
    l.set_visible(False)
    t_no_line = min(timeit.repeat(fig.canvas.draw, number=1, repeat=3))
    ax.add_line(l)
    t_invisible_line = min(timeit.repeat(fig.canvas.draw, number=1, repeat=3))
    slowdown_factor = t_invisible_line / t_no_line
    slowdown_threshold = 2
    assert slowdown_factor < slowdown_threshold