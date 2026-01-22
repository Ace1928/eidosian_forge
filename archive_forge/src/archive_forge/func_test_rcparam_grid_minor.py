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
@pytest.mark.parametrize('grid_which, major_visible, minor_visible', [('both', True, True), ('major', True, False), ('minor', False, True)])
def test_rcparam_grid_minor(grid_which, major_visible, minor_visible):
    mpl.rcParams.update({'axes.grid': True, 'axes.grid.which': grid_which})
    fig, ax = plt.subplots()
    fig.canvas.draw()
    assert all((tick.gridline.get_visible() == major_visible for tick in ax.xaxis.majorTicks))
    assert all((tick.gridline.get_visible() == minor_visible for tick in ax.xaxis.minorTicks))