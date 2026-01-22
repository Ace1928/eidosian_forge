from __future__ import annotations
from statsmodels.compat.python import lrange
from collections import defaultdict
from io import StringIO
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.base.wrapper as wrap
from statsmodels.iolib.table import SimpleTable
from statsmodels.tools.decorators import cache_readonly, deprecated_alias
from statsmodels.tools.linalg import logdet_symm
from statsmodels.tools.sm_exceptions import OutputWarning
from statsmodels.tools.validation import array_like
from statsmodels.tsa.base.tsa_model import (
import statsmodels.tsa.tsatools as tsa
from statsmodels.tsa.tsatools import duplication_matrix, unvec, vec
from statsmodels.tsa.vector_ar import output, plotting, util
from statsmodels.tsa.vector_ar.hypothesis_test_results import (
from statsmodels.tsa.vector_ar.irf import IRAnalysis
from statsmodels.tsa.vector_ar.output import VARSummary
def plot_acorr(self, nlags=10, resid=True, linewidth=8):
    """
        Plot autocorrelation of sample (endog) or residuals

        Sample (Y) or Residual autocorrelations are plotted together with the
        standard :math:`2 / \\sqrt{T}` bounds.

        Parameters
        ----------
        nlags : int
            number of lags to display (excluding 0)
        resid : bool
            If True, then the autocorrelation of the residuals is plotted
            If False, then the autocorrelation of endog is plotted.
        linewidth : int
            width of vertical bars

        Returns
        -------
        Figure
            Figure instance containing the plot.
        """
    if resid:
        acorrs = self.resid_acorr(nlags)
    else:
        acorrs = self.sample_acorr(nlags)
    bound = 2 / np.sqrt(self.nobs)
    fig = plotting.plot_full_acorr(acorrs[1:], xlabel=np.arange(1, nlags + 1), err_bound=bound, linewidth=linewidth)
    fig.suptitle('ACF plots for residuals with $2 / \\sqrt{T}$ bounds ')
    return fig