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
@pytest.mark.parametrize('left, center', [('left', ''), ('', 'center'), ('left', 'center')], ids=['left title moved', 'center title kept', 'both titles aligned'])
def test_title_above_offset(left, center):
    mpl.rcParams['axes.titley'] = None
    fig, ax = plt.subplots()
    ax.set_ylim(100000000000.0)
    ax.set_title(left, loc='left')
    ax.set_title(center)
    fig.draw_without_rendering()
    if left and (not center):
        assert ax._left_title.get_position()[1] > 1.0
    elif not left and center:
        assert ax.title.get_position()[1] == 1.0
    else:
        yleft = ax._left_title.get_position()[1]
        ycenter = ax.title.get_position()[1]
        assert yleft > 1.0
        assert ycenter == yleft