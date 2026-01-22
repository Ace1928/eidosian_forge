import importlib
import warnings
from typing import Any, Dict
import matplotlib as mpl
import numpy as np
import packaging
from matplotlib.colors import to_hex
from scipy.stats import mode, rankdata
from scipy.interpolate import CubicSpline
from ..rcparams import rcParams
from ..stats.density_utils import kde
from ..stats import hdi
def plot_point_interval(ax, values, point_estimate, hdi_prob, quartiles, linewidth, markersize, markercolor, marker, rotated, intervalcolor, backend='matplotlib'):
    """Plot point intervals.

    Translates the data and represents them as point and interval summaries.

    Parameters
    ----------
    ax : axes
        Matplotlib axes
    values : array-like
        Values to plot
    point_estimate : str
        Plot point estimate per variable.
    linewidth : int
        Line width throughout.
    quartiles : bool
        If True then the quartile interval will be plotted with the HDI.
    markersize : int
        Markersize throughout.
    markercolor: string
        Color of the marker.
    marker: string
        Shape of the marker.
    hdi_prob : float
        Valid only when point_interval is True. Plots HDI for chosen percentage of density.
    rotated : bool
        Whether to rotate the dot plot by 90 degrees.
    intervalcolor : string
        Color of the interval.
    backend : string, optional
        Matplotlib or Bokeh.
    """
    endpoint = (1 - hdi_prob) / 2
    if quartiles:
        qlist_interval = [endpoint, 0.25, 0.75, 1 - endpoint]
    else:
        qlist_interval = [endpoint, 1 - endpoint]
    quantiles_interval = np.quantile(values, qlist_interval)
    quantiles_interval[0], quantiles_interval[-1] = hdi(values.flatten(), hdi_prob, multimodal=False)
    mid = len(quantiles_interval) // 2
    param_iter = zip(np.linspace(2 * linewidth, linewidth, mid, endpoint=True)[-1::-1], range(mid))
    if backend == 'matplotlib':
        for width, j in param_iter:
            if rotated:
                ax.vlines(0, quantiles_interval[j], quantiles_interval[-(j + 1)], linewidth=width, color=intervalcolor)
            else:
                ax.hlines(0, quantiles_interval[j], quantiles_interval[-(j + 1)], linewidth=width, color=intervalcolor)
        if point_estimate:
            point_value = calculate_point_estimate(point_estimate, values)
            if rotated:
                ax.plot(0, point_value, marker, markersize=markersize, color=markercolor)
            else:
                ax.plot(point_value, 0, marker, markersize=markersize, color=markercolor)
    else:
        for width, j in param_iter:
            if rotated:
                ax.line([0, 0], [quantiles_interval[j], quantiles_interval[-(j + 1)]], line_width=width, color=intervalcolor)
            else:
                ax.line([quantiles_interval[j], quantiles_interval[-(j + 1)]], [0, 0], line_width=width, color=intervalcolor)
        if point_estimate:
            point_value = calculate_point_estimate(point_estimate, values)
            if rotated:
                ax.circle(x=0, y=point_value, size=markersize, fill_color=markercolor)
            else:
                ax.circle(x=point_value, y=0, size=markersize, fill_color=markercolor)
    return ax