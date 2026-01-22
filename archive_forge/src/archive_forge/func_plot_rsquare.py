import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import (ValueWarning,
from statsmodels.tools.validation import (string_like,
def plot_rsquare(self, ncomp=None, ax=None):
    """
        Box plots of the individual series R-square against the number of PCs.

        Parameters
        ----------
        ncomp : int, optional
            Number of components ot include in the plot.  If None, will
            plot the minimum of 10 or the number of computed components.
        ax : AxesSubplot, optional
            An axes on which to draw the graph.  If omitted, new a figure
            is created.

        Returns
        -------
        matplotlib.figure.Figure
            The handle to the figure.
        """
    import statsmodels.graphics.utils as gutils
    fig, ax = gutils.create_mpl_ax(ax)
    ncomp = 10 if ncomp is None else ncomp
    ncomp = min(ncomp, self._ncomp)
    r2s = 1.0 - self._ess_indiv / self._tss_indiv
    r2s = r2s[1:]
    r2s = r2s[:ncomp]
    ax.boxplot(r2s.T)
    ax.set_title('Individual Input $R^2$')
    ax.set_ylabel('$R^2$')
    ax.set_xlabel('Number of Included Principal Components')
    return fig