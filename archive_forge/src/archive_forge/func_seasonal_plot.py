from statsmodels.compat.pandas import deprecate_kwarg
import calendar
import numpy as np
import pandas as pd
from statsmodels.graphics import utils
from statsmodels.tools.validation import array_like
from statsmodels.tsa.stattools import acf, pacf, ccf
def seasonal_plot(grouped_x, xticklabels, ylabel=None, ax=None):
    """
    Consider using one of month_plot or quarter_plot unless you need
    irregular plotting.

    Parameters
    ----------
    grouped_x : iterable of DataFrames
        Should be a GroupBy object (or similar pair of group_names and groups
        as DataFrames) with a DatetimeIndex or PeriodIndex
    xticklabels : list of str
        List of season labels, one for each group.
    ylabel : str
        Lable for y axis
    ax : AxesSubplot, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.
    """
    fig, ax = utils.create_mpl_ax(ax)
    start = 0
    ticks = []
    for season, df in grouped_x:
        df = df.copy()
        df.sort_index()
        nobs = len(df)
        x_plot = np.arange(start, start + nobs)
        ticks.append(x_plot.mean())
        ax.plot(x_plot, df.values, 'k')
        ax.hlines(df.values.mean(), x_plot[0], x_plot[-1], colors='r', linewidth=3)
        start += nobs
    ax.set_xticks(ticks)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel(ylabel)
    ax.margins(0.1, 0.05)
    return fig