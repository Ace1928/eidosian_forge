from the axis as some gridlines can never pass any axis.
import numpy as np
import matplotlib as mpl
from matplotlib import _api
import matplotlib.axes as maxes
from matplotlib.path import Path
from mpl_toolkits.axes_grid1 import mpl_axes
from .axisline_style import AxislineStyle  # noqa
from .axis_artist import AxisArtist, GridlinesCollection
def update_lim(self, axes):
    x1, x2 = axes.get_xlim()
    y1, y2 = axes.get_ylim()
    if self._old_limits != (x1, x2, y1, y2):
        self._update_grid(x1, y1, x2, y2)
        self._old_limits = (x1, x2, y1, y2)