import numpy as np
import pandas as pd
from scipy.stats.distributions import chi2, norm
from statsmodels.graphics import utils
def plot_survfunc(survfuncs, ax=None):
    """
    Plot one or more survivor functions.

    Parameters
    ----------
    survfuncs : object or array_like
        A single SurvfuncRight object, or a list or SurvfuncRight
        objects that are plotted together.

    Returns
    -------
    A figure instance on which the plot was drawn.

    Examples
    --------
    Add a legend:

    >>> import statsmodels.api as sm
    >>> from statsmodels.duration.survfunc import plot_survfunc
    >>> data = sm.datasets.get_rdataset("flchain", "survival").data
    >>> df = data.loc[data.sex == "F", :]
    >>> sf0 = sm.SurvfuncRight(df["futime"], df["death"])
    >>> sf1 = sm.SurvfuncRight(3.0 * df["futime"], df["death"])
    >>> fig = plot_survfunc([sf0, sf1])
    >>> ax = fig.get_axes()[0]
    >>> ax.set_position([0.1, 0.1, 0.64, 0.8])
    >>> ha, lb = ax.get_legend_handles_labels()
    >>> leg = fig.legend((ha[0], ha[1]), (lb[0], lb[1]), loc='center right')

    Change the line colors:

    >>> fig = plot_survfunc([sf0, sf1])
    >>> ax = fig.get_axes()[0]
    >>> ax.set_position([0.1, 0.1, 0.64, 0.8])
    >>> ha, lb = ax.get_legend_handles_labels()
    >>> ha[0].set_color('purple')
    >>> ha[1].set_color('orange')
    """
    fig, ax = utils.create_mpl_ax(ax)
    try:
        assert type(survfuncs[0]) is SurvfuncRight
    except:
        survfuncs = [survfuncs]
    for gx, sf in enumerate(survfuncs):
        surv_times = np.concatenate(([0], sf.surv_times))
        surv_prob = np.concatenate(([1], sf.surv_prob))
        mxt = max(sf.time)
        if mxt > surv_times[-1]:
            surv_times = np.concatenate((surv_times, [mxt]))
            surv_prob = np.concatenate((surv_prob, [surv_prob[-1]]))
        label = getattr(sf, 'title', 'Group %d' % (gx + 1))
        li, = ax.step(surv_times, surv_prob, '-', label=label, lw=2, where='post')
        ii = np.flatnonzero(np.logical_not(sf.status))
        ti = np.unique(sf.time[ii])
        jj = np.searchsorted(surv_times, ti) - 1
        sp = surv_prob[jj]
        ax.plot(ti, sp, '+', ms=12, color=li.get_color(), label=label + ' points')
    ax.set_ylim(0, 1.01)
    return fig