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
@check_figures_equal()
def test_step_markers(fig_test, fig_ref):
    fig_test.subplots().step([0, 1], '-o')
    fig_ref.subplots().plot([0, 0, 1], [0, 1, 1], '-o', markevery=[0, 2])