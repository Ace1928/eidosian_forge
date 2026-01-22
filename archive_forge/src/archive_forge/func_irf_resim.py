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
def irf_resim(self, orth=False, repl=1000, steps=10, seed=None, burn=100, cum=False):
    """
        Simulates impulse response function, returning an array of simulations.
        Used for Sims-Zha error band calculation.

        Parameters
        ----------
        orth : bool, default False
            Compute orthogonalized impulse response error bands
        repl : int
            number of Monte Carlo replications to perform
        steps : int, default 10
            number of impulse response periods
        signif : float (0 < signif <1)
            Significance level for error bars, defaults to 95% CI
        seed : int
            np.random.seed for replications
        burn : int
            number of initial observations to discard for simulation
        cum : bool, default False
            produce cumulative irf error bands

        Notes
        -----
        .. [*] Sims, Christoper A., and Tao Zha. 1999. "Error Bands for Impulse
           Response." Econometrica 67: 1113-1155.

        Returns
        -------
        Array of simulated impulse response functions
        """
    neqs = self.neqs
    k_ar = self.k_ar
    coefs = self.coefs
    sigma_u = self.sigma_u
    intercept = self.intercept
    nobs = self.nobs
    nobs_original = nobs + k_ar
    ma_coll = np.zeros((repl, steps + 1, neqs, neqs))

    def fill_coll(sim):
        ret = VAR(sim, exog=self.exog).fit(maxlags=k_ar, trend=self.trend)
        ret = ret.orth_ma_rep(maxn=steps) if orth else ret.ma_rep(maxn=steps)
        return ret.cumsum(axis=0) if cum else ret
    for i in range(repl):
        sim = util.varsim(coefs, intercept, sigma_u, seed=seed, steps=nobs_original + burn)
        sim = sim[burn:]
        ma_coll[i, :, :, :] = fill_coll(sim)
    return ma_coll