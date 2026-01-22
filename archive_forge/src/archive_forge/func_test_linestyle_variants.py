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
def test_linestyle_variants():
    fig, ax = plt.subplots()
    for ls in ['-', 'solid', '--', 'dashed', '-.', 'dashdot', ':', 'dotted', (0, None), (0, ()), (0, [])]:
        ax.plot(range(10), linestyle=ls)
    fig.canvas.draw()