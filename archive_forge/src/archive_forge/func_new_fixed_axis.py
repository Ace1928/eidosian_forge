from the axis as some gridlines can never pass any axis.
import numpy as np
import matplotlib as mpl
from matplotlib import _api
import matplotlib.axes as maxes
from matplotlib.path import Path
from mpl_toolkits.axes_grid1 import mpl_axes
from .axisline_style import AxislineStyle  # noqa
from .axis_artist import AxisArtist, GridlinesCollection
def new_fixed_axis(self, loc, offset=None):
    gh = self.get_grid_helper()
    axis = gh.new_fixed_axis(loc, nth_coord=None, axis_direction=None, offset=offset, axes=self)
    return axis