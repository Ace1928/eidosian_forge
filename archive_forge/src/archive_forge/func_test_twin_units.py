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
@pytest.mark.parametrize('twin', ('x', 'y'))
def test_twin_units(twin):
    axis_name = f'{twin}axis'
    twin_func = f'twin{twin}'
    a = ['0', '1']
    b = ['a', 'b']
    fig = Figure()
    ax1 = fig.subplots()
    ax1.plot(a, b)
    assert getattr(ax1, axis_name).units is not None
    ax2 = getattr(ax1, twin_func)()
    assert getattr(ax2, axis_name).units is not None
    assert getattr(ax2, axis_name).units is getattr(ax1, axis_name).units