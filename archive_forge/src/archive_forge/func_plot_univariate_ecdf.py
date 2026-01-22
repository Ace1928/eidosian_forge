from numbers import Number
from functools import partial
import math
import textwrap
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as tx
from matplotlib.cbook import normalize_kwargs
from matplotlib.colors import to_rgba
from matplotlib.collections import LineCollection
from ._base import VectorPlotter
from ._statistics import ECDF, Histogram, KDE
from ._stats.counting import Hist
from .axisgrid import (
from .utils import (
from .palettes import color_palette
from .external import husl
from .external.kde import gaussian_kde
from ._docstrings import (
important parameter. Misspecification of the bandwidth can produce a
def plot_univariate_ecdf(self, estimate_kws, legend, **plot_kws):
    estimator = ECDF(**estimate_kws)
    drawstyles = dict(x='steps-post', y='steps-pre')
    plot_kws['drawstyle'] = drawstyles[self.data_variable]
    for sub_vars, sub_data in self.iter_data('hue', reverse=True, from_comp_data=True):
        if sub_data.empty:
            continue
        observations = sub_data[self.data_variable]
        weights = sub_data.get('weights', None)
        stat, vals = estimator(observations, weights=weights)
        artist_kws = plot_kws.copy()
        if 'hue' in self.variables:
            artist_kws['color'] = self._hue_map(sub_vars['hue'])
        ax = self._get_axes(sub_vars)
        _, inv = _get_transform_functions(ax, self.data_variable)
        vals = inv(vals)
        if isinstance(inv.__self__, mpl.scale.LogTransform):
            vals[0] = -np.inf
        if self.data_variable == 'x':
            plot_args = (vals, stat)
            stat_variable = 'y'
        else:
            plot_args = (stat, vals)
            stat_variable = 'x'
        if estimator.stat == 'count':
            top_edge = len(observations)
        else:
            top_edge = 1
        artist, = ax.plot(*plot_args, **artist_kws)
        sticky_edges = getattr(artist.sticky_edges, stat_variable)
        sticky_edges[:] = (0, top_edge)
    ax = self.ax if self.ax is not None else self.facets.axes.flat[0]
    stat = estimator.stat.capitalize()
    default_x = default_y = ''
    if self.data_variable == 'x':
        default_y = stat
    if self.data_variable == 'y':
        default_x = stat
    self._add_axis_labels(ax, default_x, default_y)
    if 'hue' in self.variables and legend:
        artist = partial(mpl.lines.Line2D, [], [])
        alpha = plot_kws.get('alpha', 1)
        ax_obj = self.ax if self.ax is not None else self.facets
        self._add_legend(ax_obj, artist, False, False, None, alpha, plot_kws, {})