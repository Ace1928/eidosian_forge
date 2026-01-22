import functools
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
import matplotlib.patches as mpatches
from matplotlib.path import Path
from mpl_toolkits.axes_grid1.parasite_axes import host_axes_class_factory
from . import axislines, grid_helper_curvelinear
from .axis_artist import AxisArtist
from .grid_finder import ExtremeFinderSimple
def trf_xy(x, y):
    trf = grid_finder.get_transform() + axes.transData
    return trf.transform(np.column_stack(np.broadcast_arrays(x, y))).T