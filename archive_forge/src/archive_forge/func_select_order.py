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
def select_order(self, maxlags=None, trend='c'):
    """
        Compute lag order selections based on each of the available information
        criteria

        Parameters
        ----------
        maxlags : int
            if None, defaults to 12 * (nobs/100.)**(1./4)
        trend : str {"n", "c", "ct", "ctt"}
            * "n" - no deterministic terms
            * "c" - constant term
            * "ct" - constant and linear term
            * "ctt" - constant, linear, and quadratic term

        Returns
        -------
        selections : LagOrderResults
        """
    ntrend = len(trend) if trend.startswith('c') else 0
    max_estimable = (self.n_totobs - self.neqs - ntrend) // (1 + self.neqs)
    if maxlags is None:
        maxlags = int(round(12 * (len(self.endog) / 100.0) ** (1 / 4.0)))
        maxlags = min(maxlags, max_estimable)
    elif maxlags > max_estimable:
        raise ValueError('maxlags is too large for the number of observations and the number of equations. The largest model cannot be estimated.')
    ics = defaultdict(list)
    p_min = 0 if self.exog is not None or trend != 'n' else 1
    for p in range(p_min, maxlags + 1):
        result = self._estimate_var(p, offset=maxlags - p, trend=trend)
        for k, v in result.info_criteria.items():
            ics[k].append(v)
    selected_orders = {k: np.array(v).argmin() + p_min for k, v in ics.items()}
    return LagOrderResults(ics, selected_orders, vecm=False)