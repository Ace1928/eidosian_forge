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
@pytest.mark.parametrize('colors', [('0.5',), ('tab:orange', 'tab:pink', 'tab:cyan', 'bLacK'), ('red', (0, 1, 0), None, (1, 0, 1, 0.5))])
def test_eventplot_colors(colors):
    """Test the *colors* parameter of eventplot. Inspired by issue #8193."""
    data = [[0], [1], [2], [3]]
    expected = [c if c is not None else 'C0' for c in colors]
    if len(expected) == 1:
        expected = expected[0]
    expected = np.broadcast_to(mcolors.to_rgba_array(expected), (len(data), 4))
    fig, ax = plt.subplots()
    if len(colors) == 1:
        colors = colors[0]
    collections = ax.eventplot(data, colors=colors)
    for coll, color in zip(collections, expected):
        assert_allclose(coll.get_color(), color)