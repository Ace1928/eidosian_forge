from collections.abc import Iterable
import copy  # check if needed when dropping python 2.7
import numpy as np
from scipy import optimize
import pandas as pd
import statsmodels.base.wrapper as wrap
from statsmodels.discrete.discrete_model import Logit
from statsmodels.genmod.generalized_linear_model import (
import statsmodels.regression.linear_model as lm
from statsmodels.tools.sm_exceptions import (PerfectSeparationError,
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.linalg import matrix_sqrt
from statsmodels.base._penalized import PenalizedMixin
from statsmodels.gam.gam_penalties import MultivariateGamPenalty
from statsmodels.gam.gam_cross_validation.gam_cross_validation import (
from statsmodels.gam.gam_cross_validation.cross_validators import KFold
def penalized_wls(endog, exog, penalty_matrix, weights):
    """weighted least squares with quadratic penalty

    Parameters
    ----------
    endog : ndarray
        response or endogenous variable
    exog : ndarray
        design matrix, matrix of exogenous or explanatory variables
    penalty_matrix : ndarray, 2-Dim square
        penality matrix for quadratic penalization. Note, the penalty_matrix
        is multiplied by two to match non-pirls fitting methods.
    weights : ndarray
        weights for WLS

    Returns
    -------
    results : Results instance of WLS
    """
    y, x, s = (endog, exog, penalty_matrix)
    aug_y, aug_x, aug_weights = make_augmented_matrix(y, x, 2 * s, weights)
    wls_results = lm.WLS(aug_y, aug_x, aug_weights).fit()
    wls_results.params = wls_results.params.ravel()
    return wls_results