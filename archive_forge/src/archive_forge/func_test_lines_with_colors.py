import contextlib
from collections import namedtuple
import datetime
from decimal import Decimal
from functools import partial
import inspect
import io
from itertools import product
import platform
from types import SimpleNamespace
import dateutil.tz
import numpy as np
from numpy import ma
from cycler import cycler
import pytest
import matplotlib
import matplotlib as mpl
from matplotlib import rc_context, patheffects
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.font_manager as mfont_manager
import matplotlib.markers as mmarkers
import matplotlib.patches as mpatches
import matplotlib.path as mpath
from matplotlib.projections.geo import HammerAxes
from matplotlib.projections.polar import PolarAxes
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import mpl_toolkits.axisartist as AA  # type: ignore
from numpy.testing import (
from matplotlib.testing.decorators import (
@pytest.mark.parametrize('data', [[1, 2, 3, np.nan, 5], np.ma.masked_equal([1, 2, 3, 4, 5], 4)])
@check_figures_equal(extensions=['png'])
def test_lines_with_colors(fig_test, fig_ref, data):
    test_colors = ['red', 'green', 'blue', 'purple', 'orange']
    fig_test.add_subplot(2, 1, 1).vlines(data, 0, 1, colors=test_colors, linewidth=5)
    fig_test.add_subplot(2, 1, 2).hlines(data, 0, 1, colors=test_colors, linewidth=5)
    expect_xy = [1, 2, 3, 5]
    expect_color = ['red', 'green', 'blue', 'orange']
    fig_ref.add_subplot(2, 1, 1).vlines(expect_xy, 0, 1, colors=expect_color, linewidth=5)
    fig_ref.add_subplot(2, 1, 2).hlines(expect_xy, 0, 1, colors=expect_color, linewidth=5)