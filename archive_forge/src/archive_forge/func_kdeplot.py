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
def kdeplot(data=None, *, x=None, y=None, hue=None, weights=None, palette=None, hue_order=None, hue_norm=None, color=None, fill=None, multiple='layer', common_norm=True, common_grid=False, cumulative=False, bw_method='scott', bw_adjust=1, warn_singular=True, log_scale=None, levels=10, thresh=0.05, gridsize=200, cut=3, clip=None, legend=True, cbar=False, cbar_ax=None, cbar_kws=None, ax=None, **kwargs):
    if 'data2' in kwargs:
        msg = '`data2` has been removed (replaced by `y`); please update your code.'
        raise TypeError(msg)
    vertical = kwargs.pop('vertical', None)
    if vertical is not None:
        if vertical:
            action_taken = 'assigning data to `y`.'
            if x is None:
                data, y = (y, data)
            else:
                x, y = (y, x)
        else:
            action_taken = 'assigning data to `x`.'
        msg = textwrap.dedent(f'\n\n        The `vertical` parameter is deprecated; {action_taken}\n        This will become an error in seaborn v0.14.0; please update your code.\n        ')
        warnings.warn(msg, UserWarning, stacklevel=2)
    bw = kwargs.pop('bw', None)
    if bw is not None:
        msg = textwrap.dedent(f'\n\n        The `bw` parameter is deprecated in favor of `bw_method` and `bw_adjust`.\n        Setting `bw_method={bw}`, but please see the docs for the new parameters\n        and update your code. This will become an error in seaborn v0.14.0.\n        ')
        warnings.warn(msg, UserWarning, stacklevel=2)
        bw_method = bw
    if kwargs.pop('kernel', None) is not None:
        msg = textwrap.dedent('\n\n        Support for alternate kernels has been removed; using Gaussian kernel.\n        This will become an error in seaborn v0.14.0; please update your code.\n        ')
        warnings.warn(msg, UserWarning, stacklevel=2)
    shade_lowest = kwargs.pop('shade_lowest', None)
    if shade_lowest is not None:
        if shade_lowest:
            thresh = 0
        msg = textwrap.dedent(f'\n\n        `shade_lowest` has been replaced by `thresh`; setting `thresh={thresh}.\n        This will become an error in seaborn v0.14.0; please update your code.\n        ')
        warnings.warn(msg, UserWarning, stacklevel=2)
    shade = kwargs.pop('shade', None)
    if shade is not None:
        fill = shade
        msg = textwrap.dedent(f'\n\n        `shade` is now deprecated in favor of `fill`; setting `fill={shade}`.\n        This will become an error in seaborn v0.14.0; please update your code.\n        ')
        warnings.warn(msg, FutureWarning, stacklevel=2)
    levels = kwargs.pop('n_levels', levels)
    p = _DistributionPlotter(data=data, variables=dict(x=x, y=y, hue=hue, weights=weights))
    p.map_hue(palette=palette, order=hue_order, norm=hue_norm)
    if ax is None:
        ax = plt.gca()
    p._attach(ax, allowed_types=['numeric', 'datetime'], log_scale=log_scale)
    method = ax.fill_between if fill else ax.plot
    color = _default_color(method, hue, color, kwargs)
    if not p.has_xy_data:
        return ax
    estimate_kws = dict(bw_method=bw_method, bw_adjust=bw_adjust, gridsize=gridsize, cut=cut, clip=clip, cumulative=cumulative)
    if p.univariate:
        plot_kws = kwargs.copy()
        p.plot_univariate_density(multiple=multiple, common_norm=common_norm, common_grid=common_grid, fill=fill, color=color, legend=legend, warn_singular=warn_singular, estimate_kws=estimate_kws, **plot_kws)
    else:
        p.plot_bivariate_density(common_norm=common_norm, fill=fill, levels=levels, thresh=thresh, legend=legend, color=color, warn_singular=warn_singular, cbar=cbar, cbar_ax=cbar_ax, cbar_kws=cbar_kws, estimate_kws=estimate_kws, **kwargs)
    return ax