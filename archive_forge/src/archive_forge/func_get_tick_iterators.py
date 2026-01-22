from the axis as some gridlines can never pass any axis.
import numpy as np
import matplotlib as mpl
from matplotlib import _api
import matplotlib.axes as maxes
from matplotlib.path import Path
from mpl_toolkits.axes_grid1 import mpl_axes
from .axisline_style import AxislineStyle  # noqa
from .axis_artist import AxisArtist, GridlinesCollection
def get_tick_iterators(self, axes):
    """tick_loc, tick_angle, tick_label"""
    if self.nth_coord == 0:
        angle_normal, angle_tangent = (90, 0)
    else:
        angle_normal, angle_tangent = (0, 90)
    major = self.axis.major
    major_locs = major.locator()
    major_labels = major.formatter.format_ticks(major_locs)
    minor = self.axis.minor
    minor_locs = minor.locator()
    minor_labels = minor.formatter.format_ticks(minor_locs)
    data_to_axes = axes.transData - axes.transAxes

    def _f(locs, labels):
        for loc, label in zip(locs, labels):
            c = self._to_xy(loc, const=self._value)
            c1, c2 = data_to_axes.transform(c)
            if 0 <= c1 <= 1 and 0 <= c2 <= 1:
                yield (c, angle_normal, angle_tangent, label)
    return (_f(major_locs, major_labels), _f(minor_locs, minor_labels))