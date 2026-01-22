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
def displot(data=None, *, x=None, y=None, hue=None, row=None, col=None, weights=None, kind='hist', rug=False, rug_kws=None, log_scale=None, legend=True, palette=None, hue_order=None, hue_norm=None, color=None, col_wrap=None, row_order=None, col_order=None, height=5, aspect=1, facet_kws=None, **kwargs):
    p = _DistributionPlotter(data=data, variables=dict(x=x, y=y, hue=hue, weights=weights, row=row, col=col))
    p.map_hue(palette=palette, order=hue_order, norm=hue_norm)
    _check_argument('kind', ['hist', 'kde', 'ecdf'], kind)
    if 'ax' in kwargs:
        msg = '`displot` is a figure-level function and does not accept the ax= parameter. You may wish to try {}plot.'.format(kind)
        warnings.warn(msg, UserWarning)
        kwargs.pop('ax')
    for var in ['row', 'col']:
        if var in p.variables and p.variables[var] is None:
            p.variables[var] = f'_{var}_'
    grid_data = p.plot_data.rename(columns=p.variables)
    grid_data = grid_data.loc[:, ~grid_data.columns.duplicated()]
    col_name = p.variables.get('col')
    row_name = p.variables.get('row')
    if facet_kws is None:
        facet_kws = {}
    g = FacetGrid(data=grid_data, row=row_name, col=col_name, col_wrap=col_wrap, row_order=row_order, col_order=col_order, height=height, aspect=aspect, **facet_kws)
    if kind == 'kde':
        allowed_types = ['numeric', 'datetime']
    else:
        allowed_types = None
    p._attach(g, allowed_types=allowed_types, log_scale=log_scale)
    if not p.has_xy_data:
        return g
    if color is None and hue is None:
        color = 'C0'
    kwargs['legend'] = legend
    if kind == 'hist':
        hist_kws = kwargs.copy()
        estimate_defaults = {}
        _assign_default_kwargs(estimate_defaults, Histogram.__init__, histplot)
        estimate_kws = {}
        for key, default_val in estimate_defaults.items():
            estimate_kws[key] = hist_kws.pop(key, default_val)
        if estimate_kws['discrete'] is None:
            estimate_kws['discrete'] = p._default_discrete()
        hist_kws['estimate_kws'] = estimate_kws
        hist_kws.setdefault('color', color)
        if p.univariate:
            _assign_default_kwargs(hist_kws, p.plot_univariate_histogram, histplot)
            p.plot_univariate_histogram(**hist_kws)
        else:
            _assign_default_kwargs(hist_kws, p.plot_bivariate_histogram, histplot)
            p.plot_bivariate_histogram(**hist_kws)
    elif kind == 'kde':
        kde_kws = kwargs.copy()
        estimate_defaults = {}
        _assign_default_kwargs(estimate_defaults, KDE.__init__, kdeplot)
        estimate_kws = {}
        for key, default_val in estimate_defaults.items():
            estimate_kws[key] = kde_kws.pop(key, default_val)
        kde_kws['estimate_kws'] = estimate_kws
        kde_kws['color'] = color
        if p.univariate:
            _assign_default_kwargs(kde_kws, p.plot_univariate_density, kdeplot)
            p.plot_univariate_density(**kde_kws)
        else:
            _assign_default_kwargs(kde_kws, p.plot_bivariate_density, kdeplot)
            p.plot_bivariate_density(**kde_kws)
    elif kind == 'ecdf':
        ecdf_kws = kwargs.copy()
        estimate_kws = {}
        estimate_defaults = {}
        _assign_default_kwargs(estimate_defaults, ECDF.__init__, ecdfplot)
        for key, default_val in estimate_defaults.items():
            estimate_kws[key] = ecdf_kws.pop(key, default_val)
        ecdf_kws['estimate_kws'] = estimate_kws
        ecdf_kws['color'] = color
        if p.univariate:
            _assign_default_kwargs(ecdf_kws, p.plot_univariate_ecdf, ecdfplot)
            p.plot_univariate_ecdf(**ecdf_kws)
        else:
            raise NotImplementedError('Bivariate ECDF plots are not implemented')
    if rug:
        if rug_kws is None:
            rug_kws = {}
        _assign_default_kwargs(rug_kws, p.plot_rug, rugplot)
        rug_kws['legend'] = False
        if color is not None:
            rug_kws['color'] = color
        p.plot_rug(**rug_kws)
    g.set_axis_labels(x_var=p.variables.get('x', g.axes.flat[0].get_xlabel()), y_var=p.variables.get('y', g.axes.flat[0].get_ylabel()))
    g.set_titles()
    g.tight_layout()
    if data is not None and (x is not None or y is not None):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        g.data = pd.merge(data, g.data[g.data.columns.difference(data.columns)], left_index=True, right_index=True)
    else:
        wide_cols = {k: f'_{k}_' if v is None else v for k, v in p.variables.items()}
        g.data = p.plot_data.rename(columns=wide_cols)
    return g