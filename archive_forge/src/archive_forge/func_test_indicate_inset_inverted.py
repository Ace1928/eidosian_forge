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
@pytest.mark.parametrize('x_inverted', [False, True])
@pytest.mark.parametrize('y_inverted', [False, True])
def test_indicate_inset_inverted(x_inverted, y_inverted):
    """
    Test that the inset lines are correctly located with inverted data axes.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2)
    x = np.arange(10)
    ax1.plot(x, x, 'o')
    if x_inverted:
        ax1.invert_xaxis()
    if y_inverted:
        ax1.invert_yaxis()
    rect, bounds = ax1.indicate_inset([2, 2, 5, 4], ax2)
    lower_left, upper_left, lower_right, upper_right = bounds
    sign_x = -1 if x_inverted else 1
    sign_y = -1 if y_inverted else 1
    assert sign_x * (lower_right.xy2[0] - lower_left.xy2[0]) > 0
    assert sign_x * (upper_right.xy2[0] - upper_left.xy2[0]) > 0
    assert sign_y * (upper_left.xy2[1] - lower_left.xy2[1]) > 0
    assert sign_y * (upper_right.xy2[1] - lower_right.xy2[1]) > 0