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
def test_stem_args():
    """Test that stem() correctly identifies x and y values."""

    def _assert_equal(stem_container, expected):
        x, y = map(list, stem_container.markerline.get_data())
        assert x == expected[0]
        assert y == expected[1]
    fig, ax = plt.subplots()
    x = [1, 3, 5]
    y = [9, 8, 7]
    _assert_equal(ax.stem(y), expected=([0, 1, 2], y))
    _assert_equal(ax.stem(x, y), expected=(x, y))
    _assert_equal(ax.stem(x, y, linefmt='r--'), expected=(x, y))
    _assert_equal(ax.stem(x, y, 'r--'), expected=(x, y))
    _assert_equal(ax.stem(x, y, linefmt='r--', basefmt='b--'), expected=(x, y))
    _assert_equal(ax.stem(y, linefmt='r--'), expected=([0, 1, 2], y))
    _assert_equal(ax.stem(y, 'r--'), expected=([0, 1, 2], y))