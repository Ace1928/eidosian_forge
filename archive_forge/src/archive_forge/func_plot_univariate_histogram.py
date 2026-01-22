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
def plot_univariate_histogram(self, multiple, element, fill, common_norm, common_bins, shrink, kde, kde_kws, color, legend, line_kws, estimate_kws, **plot_kws):
    kde_kws = {} if kde_kws is None else kde_kws.copy()
    line_kws = {} if line_kws is None else line_kws.copy()
    estimate_kws = {} if estimate_kws is None else estimate_kws.copy()
    _check_argument('multiple', ['layer', 'stack', 'fill', 'dodge'], multiple)
    _check_argument('element', ['bars', 'step', 'poly'], element)
    auto_bins_with_weights = 'weights' in self.variables and estimate_kws['bins'] == 'auto' and (estimate_kws['binwidth'] is None) and (not estimate_kws['discrete'])
    if auto_bins_with_weights:
        msg = "`bins` cannot be 'auto' when using weights. Setting `bins=10`, but you will likely want to adjust."
        warnings.warn(msg, UserWarning)
        estimate_kws['bins'] = 10
    if estimate_kws['stat'] == 'count':
        common_norm = False
    orient = self.data_variable
    estimator = Hist(**estimate_kws)
    histograms = {}
    all_data = self.comp_data.dropna()
    all_weights = all_data.get('weights', None)
    multiple_histograms = set(self.variables) - {'x', 'y'}
    if multiple_histograms:
        if common_bins:
            bin_kws = estimator._define_bin_params(all_data, orient, None)
    else:
        common_norm = False
    if common_norm and all_weights is not None:
        whole_weight = all_weights.sum()
    else:
        whole_weight = len(all_data)
    if kde:
        kde_kws.setdefault('cut', 0)
        kde_kws['cumulative'] = estimate_kws['cumulative']
        densities = self._compute_univariate_density(self.data_variable, common_norm, common_bins, kde_kws, warn_singular=False)
    for sub_vars, sub_data in self.iter_data('hue', from_comp_data=True):
        key = tuple(sub_vars.items())
        orient = self.data_variable
        if 'weights' in self.variables:
            sub_data['weight'] = sub_data.pop('weights')
            part_weight = sub_data['weight'].sum()
        else:
            part_weight = len(sub_data)
        if not (multiple_histograms and common_bins):
            bin_kws = estimator._define_bin_params(sub_data, orient, None)
        res = estimator._normalize(estimator._eval(sub_data, orient, bin_kws))
        heights = res[estimator.stat].to_numpy()
        widths = res['space'].to_numpy()
        edges = res[orient].to_numpy() - widths / 2
        if kde and key in densities:
            density = densities[key]
            if estimator.cumulative:
                hist_norm = heights.max()
            else:
                hist_norm = (heights * widths).sum()
            densities[key] *= hist_norm
        ax = self._get_axes(sub_vars)
        _, inv = _get_transform_functions(ax, self.data_variable)
        widths = inv(edges + widths) - inv(edges)
        edges = inv(edges)
        edges = edges + (1 - shrink) / 2 * widths
        widths *= shrink
        index = pd.MultiIndex.from_arrays([pd.Index(edges, name='edges'), pd.Index(widths, name='widths')])
        hist = pd.Series(heights, index=index, name='heights')
        if common_norm:
            hist *= part_weight / whole_weight
        histograms[key] = hist
    histograms, baselines = self._resolve_multiple(histograms, multiple)
    if kde:
        densities, _ = self._resolve_multiple(densities, None if multiple == 'dodge' else multiple)
    sticky_stat = (0, 1) if multiple == 'fill' else (0, np.inf)
    if multiple == 'fill':
        bin_vals = histograms.index.to_frame()
        edges = bin_vals['edges']
        widths = bin_vals['widths']
        sticky_data = (edges.min(), edges.max() + widths.loc[edges.idxmax()])
    else:
        sticky_data = []
    if fill:
        if 'hue' in self.variables and multiple == 'layer':
            default_alpha = 0.5 if element == 'bars' else 0.25
        elif kde:
            default_alpha = 0.5
        else:
            default_alpha = 0.75
    else:
        default_alpha = 1
    alpha = plot_kws.pop('alpha', default_alpha)
    hist_artists = []
    for sub_vars, _ in self.iter_data('hue', reverse=True):
        key = tuple(sub_vars.items())
        hist = histograms[key].rename('heights').reset_index()
        bottom = np.asarray(baselines[key])
        ax = self._get_axes(sub_vars)
        if 'hue' in self.variables:
            sub_color = self._hue_map(sub_vars['hue'])
        else:
            sub_color = color
        artist_kws = self._artist_kws(plot_kws, fill, element, multiple, sub_color, alpha)
        if element == 'bars':
            plot_func = ax.bar if self.data_variable == 'x' else ax.barh
            artists = plot_func(hist['edges'], hist['heights'] - bottom, hist['widths'], bottom, align='edge', **artist_kws)
            for bar in artists:
                if self.data_variable == 'x':
                    bar.sticky_edges.x[:] = sticky_data
                    bar.sticky_edges.y[:] = sticky_stat
                else:
                    bar.sticky_edges.x[:] = sticky_stat
                    bar.sticky_edges.y[:] = sticky_data
            hist_artists.extend(artists)
        else:
            if element == 'step':
                final = hist.iloc[-1]
                x = np.append(hist['edges'], final['edges'] + final['widths'])
                y = np.append(hist['heights'], final['heights'])
                b = np.append(bottom, bottom[-1])
                if self.data_variable == 'x':
                    step = 'post'
                    drawstyle = 'steps-post'
                else:
                    step = 'post'
                    drawstyle = 'steps-pre'
            elif element == 'poly':
                x = hist['edges'] + hist['widths'] / 2
                y = hist['heights']
                b = bottom
                step = None
                drawstyle = None
            if self.data_variable == 'x':
                if fill:
                    artist = ax.fill_between(x, b, y, step=step, **artist_kws)
                else:
                    artist, = ax.plot(x, y, drawstyle=drawstyle, **artist_kws)
                artist.sticky_edges.x[:] = sticky_data
                artist.sticky_edges.y[:] = sticky_stat
            else:
                if fill:
                    artist = ax.fill_betweenx(x, b, y, step=step, **artist_kws)
                else:
                    artist, = ax.plot(y, x, drawstyle=drawstyle, **artist_kws)
                artist.sticky_edges.x[:] = sticky_stat
                artist.sticky_edges.y[:] = sticky_data
            hist_artists.append(artist)
        if kde:
            try:
                density = densities[key]
            except KeyError:
                continue
            support = density.index
            if 'x' in self.variables:
                line_args = (support, density)
                sticky_x, sticky_y = (None, (0, np.inf))
            else:
                line_args = (density, support)
                sticky_x, sticky_y = ((0, np.inf), None)
            line_kws['color'] = to_rgba(sub_color, 1)
            line, = ax.plot(*line_args, **line_kws)
            if sticky_x is not None:
                line.sticky_edges.x[:] = sticky_x
            if sticky_y is not None:
                line.sticky_edges.y[:] = sticky_y
    if element == 'bars' and 'linewidth' not in plot_kws:
        hist_metadata = pd.concat([h.index.to_frame() for _, h in histograms.items()]).reset_index(drop=True)
        thin_bar_idx = hist_metadata['widths'].idxmin()
        binwidth = hist_metadata.loc[thin_bar_idx, 'widths']
        left_edge = hist_metadata.loc[thin_bar_idx, 'edges']
        default_linewidth = math.inf
        for sub_vars, _ in self.iter_data():
            ax = self._get_axes(sub_vars)
            ax.autoscale_view()
            pts_x, pts_y = 72 / ax.figure.dpi * abs(ax.transData.transform([left_edge + binwidth] * 2) - ax.transData.transform([left_edge] * 2))
            if self.data_variable == 'x':
                binwidth_points = pts_x
            else:
                binwidth_points = pts_y
            default_linewidth = min(0.1 * binwidth_points, default_linewidth)
        for bar in hist_artists:
            max_linewidth = bar.get_linewidth()
            if not fill:
                max_linewidth *= 1.5
            linewidth = min(default_linewidth, max_linewidth)
            if not fill:
                min_linewidth = 0.5
                linewidth = max(linewidth, min_linewidth)
            bar.set_linewidth(linewidth)
    ax = self.ax if self.ax is not None else self.facets.axes.flat[0]
    default_x = default_y = ''
    if self.data_variable == 'x':
        default_y = estimator.stat.capitalize()
    if self.data_variable == 'y':
        default_x = estimator.stat.capitalize()
    self._add_axis_labels(ax, default_x, default_y)
    if 'hue' in self.variables and legend:
        if fill or element == 'bars':
            artist = partial(mpl.patches.Patch)
        else:
            artist = partial(mpl.lines.Line2D, [], [])
        ax_obj = self.ax if self.ax is not None else self.facets
        self._add_legend(ax_obj, artist, fill, element, multiple, alpha, plot_kws, {})