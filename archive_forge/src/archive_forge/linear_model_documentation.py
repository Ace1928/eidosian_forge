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

        Compute the confidence interval using Empirical Likelihood.

        Parameters
        ----------
        param_num : float
            The parameter for which the confidence interval is desired.
        sig : float
            The significance level.  Default is 0.05.
        upper_bound : float
            The maximum value the upper limit can be.  Default is the
            99.9% confidence value under OLS assumptions.
        lower_bound : float
            The minimum value the lower limit can be.  Default is the 99.9%
            confidence value under OLS assumptions.
        method : str
            Can either be 'nm' for Nelder-Mead or 'powell' for Powell.  The
            optimization method that optimizes over nuisance parameters.
            The default is 'nm'.
        stochastic_exog : bool
            When True, the exogenous variables are assumed to be stochastic.
            When the regressors are nonstochastic, moment conditions are
            placed on the exogenous variables.  Confidence intervals for
            stochastic regressors are at least as large as non-stochastic
            regressors.  The default is True.

        Returns
        -------
        lowerl : float
            The lower bound of the confidence interval.
        upperl : float
            The upper bound of the confidence interval.

        See Also
        --------
        el_test : Test parameters using Empirical Likelihood.

        Notes
        -----
        This function uses brentq to find the value of beta where
        test_beta([beta], param_num)[1] is equal to the critical value.

        The function returns the results of each iteration of brentq at each
        value of beta.

        The current function value of the last printed optimization should be
        the critical value at the desired significance level. For alpha=.05,
        the value is 3.841459.

        To ensure optimization terminated successfully, it is suggested to do
        el_test([lower_limit], [param_num]).

        If the optimization does not terminate successfully, consider switching
        optimization algorithms.

        If optimization is still not successful, try changing the values of
        start_int_params.  If the current function value repeatedly jumps
        from a number between 0 and the critical value and a very large number
        (>50), the starting parameters of the interior minimization need
        to be changed.
        