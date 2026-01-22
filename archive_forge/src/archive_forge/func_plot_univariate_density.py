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
def plot_univariate_density(self, multiple, common_norm, common_grid, warn_singular, fill, color, legend, estimate_kws, **plot_kws):
    if fill is None:
        fill = multiple in ('stack', 'fill')
    if fill:
        artist = mpl.collections.PolyCollection
    else:
        artist = mpl.lines.Line2D
    plot_kws = normalize_kwargs(plot_kws, artist)
    _check_argument('multiple', ['layer', 'stack', 'fill'], multiple)
    subsets = bool(set(self.variables) - {'x', 'y'})
    if subsets and multiple in ('stack', 'fill'):
        common_grid = True
    densities = self._compute_univariate_density(self.data_variable, common_norm, common_grid, estimate_kws, warn_singular)
    densities, baselines = self._resolve_multiple(densities, multiple)
    sticky_density = (0, 1) if multiple == 'fill' else (0, np.inf)
    if multiple == 'fill':
        sticky_support = (densities.index.min(), densities.index.max())
    else:
        sticky_support = []
    if fill:
        if multiple == 'layer':
            default_alpha = 0.25
        else:
            default_alpha = 0.75
    else:
        default_alpha = 1
    alpha = plot_kws.pop('alpha', default_alpha)
    for sub_vars, _ in self.iter_data('hue', reverse=True):
        key = tuple(sub_vars.items())
        try:
            density = densities[key]
        except KeyError:
            continue
        support = density.index
        fill_from = baselines[key]
        ax = self._get_axes(sub_vars)
        if 'hue' in self.variables:
            sub_color = self._hue_map(sub_vars['hue'])
        else:
            sub_color = color
        artist_kws = self._artist_kws(plot_kws, fill, False, multiple, sub_color, alpha)
        if 'x' in self.variables:
            if fill:
                artist = ax.fill_between(support, fill_from, density, **artist_kws)
            else:
                artist, = ax.plot(support, density, **artist_kws)
            artist.sticky_edges.x[:] = sticky_support
            artist.sticky_edges.y[:] = sticky_density
        else:
            if fill:
                artist = ax.fill_betweenx(support, fill_from, density, **artist_kws)
            else:
                artist, = ax.plot(density, support, **artist_kws)
            artist.sticky_edges.x[:] = sticky_density
            artist.sticky_edges.y[:] = sticky_support
    ax = self.ax if self.ax is not None else self.facets.axes.flat[0]
    default_x = default_y = ''
    if self.data_variable == 'x':
        default_y = 'Density'
    if self.data_variable == 'y':
        default_x = 'Density'
    self._add_axis_labels(ax, default_x, default_y)
    if 'hue' in self.variables and legend:
        if fill:
            artist = partial(mpl.patches.Patch)
        else:
            artist = partial(mpl.lines.Line2D, [], [])
        ax_obj = self.ax if self.ax is not None else self.facets
        self._add_legend(ax_obj, artist, fill, False, multiple, alpha, plot_kws, {})