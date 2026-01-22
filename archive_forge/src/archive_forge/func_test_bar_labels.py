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
@pytest.mark.parametrize(('x', 'width', 'label', 'expected_labels', 'container_label'), [('x', 1, 'x', ['_nolegend_'], 'x'), (['a', 'b', 'c'], [10, 20, 15], ['A', 'B', 'C'], ['A', 'B', 'C'], '_nolegend_'), (['a', 'b', 'c'], [10, 20, 15], ['R', 'Y', '_nolegend_'], ['R', 'Y', '_nolegend_'], '_nolegend_'), (['a', 'b', 'c'], [10, 20, 15], 'bars', ['_nolegend_', '_nolegend_', '_nolegend_'], 'bars')])
def test_bar_labels(x, width, label, expected_labels, container_label):
    _, ax = plt.subplots()
    bar_container = ax.bar(x, width, label=label)
    bar_labels = [bar.get_label() for bar in bar_container]
    assert expected_labels == bar_labels
    assert bar_container.get_label() == container_label