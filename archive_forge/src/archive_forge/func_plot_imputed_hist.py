import pandas as pd
import numpy as np
import patsy
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.regression.linear_model import OLS
from collections import defaultdict
def plot_imputed_hist(self, col_name, ax=None, imp_hist_args=None, obs_hist_args=None, all_hist_args=None):
    """
        Display imputed values for one variable as a histogram.

        Parameters
        ----------
        col_name : str
            The name of the variable to be plotted.
        ax : AxesSubplot
            An axes on which to draw the histograms.  If not provided,
            one is created.
        imp_hist_args : dict
            Keyword arguments to be passed to pyplot.hist when
            creating the histogram for imputed values.
        obs_hist_args : dict
            Keyword arguments to be passed to pyplot.hist when
            creating the histogram for observed values.
        all_hist_args : dict
            Keyword arguments to be passed to pyplot.hist when
            creating the histogram for all values.

        Returns
        -------
        The matplotlib figure on which the histograms were drawn
        """
    from statsmodels.graphics import utils as gutils
    if imp_hist_args is None:
        imp_hist_args = {}
    if obs_hist_args is None:
        obs_hist_args = {}
    if all_hist_args is None:
        all_hist_args = {}
    if ax is None:
        fig, ax = gutils.create_mpl_ax(ax)
    else:
        fig = ax.get_figure()
    ax.set_position([0.1, 0.1, 0.7, 0.8])
    ixm = self.ix_miss[col_name]
    ixo = self.ix_obs[col_name]
    imp = self.data[col_name].iloc[ixm]
    obs = self.data[col_name].iloc[ixo]
    for di in (imp_hist_args, obs_hist_args, all_hist_args):
        if 'histtype' not in di:
            di['histtype'] = 'step'
    ha, la = ([], [])
    if len(imp) > 0:
        h = ax.hist(np.asarray(imp), **imp_hist_args)
        ha.append(h[-1][0])
        la.append('Imp')
    h1 = ax.hist(np.asarray(obs), **obs_hist_args)
    h2 = ax.hist(np.asarray(self.data[col_name]), **all_hist_args)
    ha.extend([h1[-1][0], h2[-1][0]])
    la.extend(['Obs', 'All'])
    leg = fig.legend(ha, la, loc='center right', numpoints=1)
    leg.draw_frame(False)
    ax.set_xlabel(col_name)
    ax.set_ylabel('Frequency')
    return fig