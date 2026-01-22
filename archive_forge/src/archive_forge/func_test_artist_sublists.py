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
def test_artist_sublists():
    fig, ax = plt.subplots()
    lines = [ax.plot(np.arange(i, i + 5))[0] for i in range(6)]
    col = ax.scatter(np.arange(5), np.arange(5))
    im = ax.imshow(np.zeros((5, 5)))
    patch = ax.add_patch(mpatches.Rectangle((0, 0), 5, 5))
    text = ax.text(0, 0, 'foo')
    assert list(ax.collections) == [col]
    assert list(ax.images) == [im]
    assert list(ax.lines) == lines
    assert list(ax.patches) == [patch]
    assert not ax.tables
    assert list(ax.texts) == [text]
    assert ax.lines[0] is lines[0]
    assert ax.lines[-1] is lines[-1]
    with pytest.raises(IndexError, match='out of range'):
        ax.lines[len(lines) + 1]
    assert ax.lines + [1, 2, 3] == [*lines, 1, 2, 3]
    assert [1, 2, 3] + ax.lines == [1, 2, 3, *lines]
    assert ax.lines + (1, 2, 3) == (*lines, 1, 2, 3)
    assert (1, 2, 3) + ax.lines == (1, 2, 3, *lines)
    col.remove()
    assert not ax.collections
    im.remove()
    assert not ax.images
    patch.remove()
    assert not ax.patches
    assert not ax.tables
    text.remove()
    assert not ax.texts
    for ln in ax.lines:
        ln.remove()
    assert len(ax.lines) == 0