from __future__ import annotations
from statsmodels.compat.pandas import Appender
from statsmodels.compat.python import lrange, lzip
from typing import Literal
from collections.abc import Sequence
import warnings
import numpy as np
from scipy import optimize, stats
from scipy.linalg import cholesky, toeplitz
from scipy.linalg.lapack import dtrtri
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.emplike.elregress import _ELRegOpts
from statsmodels.regression._prediction import PredictionResults
from statsmodels.tools.decorators import cache_readonly, cache_writable
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.tools import pinv_extended
from statsmodels.tools.typing import Float64Array
from statsmodels.tools.validation import bool_like, float_like, string_like
from . import _prediction as pred
def compare_f_test(self, restricted):
    """
        Use F test to test whether restricted model is correct.

        Parameters
        ----------
        restricted : Result instance
            The restricted model is assumed to be nested in the
            current model. The result instance of the restricted model
            is required to have two attributes, residual sum of
            squares, `ssr`, residual degrees of freedom, `df_resid`.

        Returns
        -------
        f_value : float
            The test statistic which has an F distribution.
        p_value : float
            The p-value of the test statistic.
        df_diff : int
            The degrees of freedom of the restriction, i.e. difference in
            df between models.

        Notes
        -----
        See mailing list discussion October 17,

        This test compares the residual sum of squares of the two
        models.  This is not a valid test, if there is unspecified
        heteroscedasticity or correlation. This method will issue a
        warning if this is detected but still return the results under
        the assumption of homoscedasticity and no autocorrelation
        (sphericity).
        """
    has_robust1 = getattr(self, 'cov_type', 'nonrobust') != 'nonrobust'
    has_robust2 = getattr(restricted, 'cov_type', 'nonrobust') != 'nonrobust'
    if has_robust1 or has_robust2:
        warnings.warn('F test for comparison is likely invalid with ' + 'robust covariance, proceeding anyway', InvalidTestWarning)
    ssr_full = self.ssr
    ssr_restr = restricted.ssr
    df_full = self.df_resid
    df_restr = restricted.df_resid
    df_diff = df_restr - df_full
    f_value = (ssr_restr - ssr_full) / df_diff / ssr_full * df_full
    p_value = stats.f.sf(f_value, df_diff, df_full)
    return (f_value, p_value, df_diff)