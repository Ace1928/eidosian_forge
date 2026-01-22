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
def distplot(a=None, bins=None, hist=True, kde=True, rug=False, fit=None, hist_kws=None, kde_kws=None, rug_kws=None, fit_kws=None, color=None, vertical=False, norm_hist=False, axlabel=None, label=None, ax=None, x=None):
    """
    DEPRECATED

    This function has been deprecated and will be removed in seaborn v0.14.0.
    It has been replaced by :func:`histplot` and :func:`displot`, two functions
    with a modern API and many more capabilities.

    For a guide to updating, please see this notebook:

    https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751

    """
    if kde and (not hist):
        axes_level_suggestion = '`kdeplot` (an axes-level function for kernel density plots)'
    else:
        axes_level_suggestion = '`histplot` (an axes-level function for histograms)'
    msg = textwrap.dedent(f'\n\n    `distplot` is a deprecated function and will be removed in seaborn v0.14.0.\n\n    Please adapt your code to use either `displot` (a figure-level function with\n    similar flexibility) or {axes_level_suggestion}.\n\n    For a guide to updating your code to use the new functions, please see\n    https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751\n    ')
    warnings.warn(msg, UserWarning, stacklevel=2)
    if ax is None:
        ax = plt.gca()
    label_ax = bool(axlabel)
    if axlabel is None and hasattr(a, 'name'):
        axlabel = a.name
        if axlabel is not None:
            label_ax = True
    if x is not None:
        a = x
    a = np.asarray(a, float)
    if a.ndim > 1:
        a = a.squeeze()
    a = remove_na(a)
    norm_hist = norm_hist or kde or fit is not None
    hist_kws = {} if hist_kws is None else hist_kws.copy()
    kde_kws = {} if kde_kws is None else kde_kws.copy()
    rug_kws = {} if rug_kws is None else rug_kws.copy()
    fit_kws = {} if fit_kws is None else fit_kws.copy()
    if color is None:
        if vertical:
            line, = ax.plot(0, a.mean())
        else:
            line, = ax.plot(a.mean(), 0)
        color = line.get_color()
        line.remove()
    if label is not None:
        if hist:
            hist_kws['label'] = label
        elif kde:
            kde_kws['label'] = label
        elif rug:
            rug_kws['label'] = label
        elif fit:
            fit_kws['label'] = label
    if hist:
        if bins is None:
            bins = min(_freedman_diaconis_bins(a), 50)
        hist_kws.setdefault('alpha', 0.4)
        hist_kws.setdefault('density', norm_hist)
        orientation = 'horizontal' if vertical else 'vertical'
        hist_color = hist_kws.pop('color', color)
        ax.hist(a, bins, orientation=orientation, color=hist_color, **hist_kws)
        if hist_color != color:
            hist_kws['color'] = hist_color
    axis = 'y' if vertical else 'x'
    if kde:
        kde_color = kde_kws.pop('color', color)
        kdeplot(**{axis: a}, ax=ax, color=kde_color, **kde_kws)
        if kde_color != color:
            kde_kws['color'] = kde_color
    if rug:
        rug_color = rug_kws.pop('color', color)
        rugplot(**{axis: a}, ax=ax, color=rug_color, **rug_kws)
        if rug_color != color:
            rug_kws['color'] = rug_color
    if fit is not None:

        def pdf(x):
            return fit.pdf(x, *params)
        fit_color = fit_kws.pop('color', '#282828')
        gridsize = fit_kws.pop('gridsize', 200)
        cut = fit_kws.pop('cut', 3)
        clip = fit_kws.pop('clip', (-np.inf, np.inf))
        bw = gaussian_kde(a).scotts_factor() * a.std(ddof=1)
        x = _kde_support(a, bw, gridsize, cut, clip)
        params = fit.fit(a)
        y = pdf(x)
        if vertical:
            x, y = (y, x)
        ax.plot(x, y, color=fit_color, **fit_kws)
        if fit_color != '#282828':
            fit_kws['color'] = fit_color
    if label_ax:
        if vertical:
            ax.set_ylabel(axlabel)
        else:
            ax.set_xlabel(axlabel)
    return ax