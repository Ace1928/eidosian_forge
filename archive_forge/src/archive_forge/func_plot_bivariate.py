import pandas as pd
import numpy as np
import patsy
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.regression.linear_model import OLS
from collections import defaultdict
def plot_bivariate(self, col1_name, col2_name, lowess_args=None, lowess_min_n=40, jitter=None, plot_points=True, ax=None):
    """
        Plot observed and imputed values for two variables.

        Displays a scatterplot of one variable against another.  The
        points are colored according to whether the values are
        observed or imputed.

        Parameters
        ----------
        col1_name : str
            The variable to be plotted on the horizontal axis.
        col2_name : str
            The variable to be plotted on the vertical axis.
        lowess_args : dictionary
            A dictionary of dictionaries, keys are 'ii', 'io', 'oi'
            and 'oo', where 'o' denotes 'observed' and 'i' denotes
            imputed.  See Notes for details.
        lowess_min_n : int
            Minimum sample size to plot a lowess fit
        jitter : float or tuple
            Standard deviation for jittering points in the plot.
            Either a single scalar applied to both axes, or a tuple
            containing x-axis jitter and y-axis jitter, respectively.
        plot_points : bool
            If True, the data points are plotted.
        ax : AxesSubplot
            Axes on which to plot, created if not provided.

        Returns
        -------
        The matplotlib figure on which the plot id drawn.
        """
    from statsmodels.graphics import utils as gutils
    from statsmodels.nonparametric.smoothers_lowess import lowess
    if lowess_args is None:
        lowess_args = {}
    if ax is None:
        fig, ax = gutils.create_mpl_ax(ax)
    else:
        fig = ax.get_figure()
    ax.set_position([0.1, 0.1, 0.7, 0.8])
    ix1i = self.ix_miss[col1_name]
    ix1o = self.ix_obs[col1_name]
    ix2i = self.ix_miss[col2_name]
    ix2o = self.ix_obs[col2_name]
    ix_ii = np.intersect1d(ix1i, ix2i)
    ix_io = np.intersect1d(ix1i, ix2o)
    ix_oi = np.intersect1d(ix1o, ix2i)
    ix_oo = np.intersect1d(ix1o, ix2o)
    vec1 = np.require(self.data[col1_name], requirements='W')
    vec2 = np.require(self.data[col2_name], requirements='W')
    if jitter is not None:
        if np.isscalar(jitter):
            jitter = (jitter, jitter)
        vec1 += jitter[0] * np.random.normal(size=len(vec1))
        vec2 += jitter[1] * np.random.normal(size=len(vec2))
    keys = ['oo', 'io', 'oi', 'ii']
    lak = {'i': 'imp', 'o': 'obs'}
    ixs = {'ii': ix_ii, 'io': ix_io, 'oi': ix_oi, 'oo': ix_oo}
    color = {'oo': 'grey', 'ii': 'red', 'io': 'orange', 'oi': 'lime'}
    if plot_points:
        for ky in keys:
            ix = ixs[ky]
            lab = lak[ky[0]] + '/' + lak[ky[1]]
            ax.plot(vec1[ix], vec2[ix], 'o', color=color[ky], label=lab, alpha=0.6)
    for ky in keys:
        ix = ixs[ky]
        if len(ix) < lowess_min_n:
            continue
        if ky in lowess_args:
            la = lowess_args[ky]
        else:
            la = {}
        ix = ixs[ky]
        lfit = lowess(vec2[ix], vec1[ix], **la)
        if plot_points:
            ax.plot(lfit[:, 0], lfit[:, 1], '-', color=color[ky], alpha=0.6, lw=4)
        else:
            lab = lak[ky[0]] + '/' + lak[ky[1]]
            ax.plot(lfit[:, 0], lfit[:, 1], '-', color=color[ky], alpha=0.6, lw=4, label=lab)
    ha, la = ax.get_legend_handles_labels()
    pad = 0.0001 if plot_points else 0.5
    leg = fig.legend(ha, la, loc='center right', numpoints=1, handletextpad=pad)
    leg.draw_frame(False)
    ax.set_xlabel(col1_name)
    ax.set_ylabel(col2_name)
    return fig