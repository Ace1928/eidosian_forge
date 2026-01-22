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
@check_figures_equal(extensions=['pdf'])
def test_2dcolor_plot(fig_test, fig_ref):
    color = np.array([0.1, 0.2, 0.3])
    axs = fig_test.subplots(5)
    axs[0].plot([1, 2], [1, 2], c=color.reshape(-1))
    with pytest.warns(match='argument looks like a single numeric RGB'):
        axs[1].scatter([1, 2], [1, 2], c=color.reshape(-1))
    axs[2].step([1, 2], [1, 2], c=color.reshape(-1))
    axs[3].hist(np.arange(10), color=color.reshape(-1))
    axs[4].bar(np.arange(10), np.arange(10), color=color.reshape(-1))
    axs = fig_ref.subplots(5)
    axs[0].plot([1, 2], [1, 2], c=color.reshape((1, -1)))
    axs[1].scatter([1, 2], [1, 2], c=color.reshape((1, -1)))
    axs[2].step([1, 2], [1, 2], c=color.reshape((1, -1)))
    axs[3].hist(np.arange(10), color=color.reshape((1, -1)))
    axs[4].bar(np.arange(10), np.arange(10), color=color.reshape((1, -1)))