from statsmodels.compat.python import lzip
from statsmodels.compat.pandas import Appender
import numpy as np
from scipy import stats
import pandas as pd
import patsy
from collections import defaultdict
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.model as base
import statsmodels.regression.linear_model as lm
import statsmodels.base.wrapper as wrap
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM, GLMResults
from statsmodels.genmod import cov_struct as cov_structs
import statsmodels.genmod.families.varfuncs as varfuncs
from statsmodels.genmod.families.links import Link
from statsmodels.tools.sm_exceptions import (ConvergenceWarning,
import warnings
from statsmodels.graphics._regressionplots_doc import (
from statsmodels.discrete.discrete_margins import (
def mean_deriv(self, exog, lin_pred):
    """
        Derivative of the expected endog with respect to the parameters.

        Parameters
        ----------
        exog : array_like
           The exogeneous data at which the derivative is computed,
           number of rows must be a multiple of `ncut`.
        lin_pred : array_like
           The values of the linear predictor, length must be multiple
           of `ncut`.

        Returns
        -------
        The derivative of the expected endog with respect to the
        parameters.
        """
    expval = np.exp(lin_pred)
    expval_m = np.reshape(expval, (len(expval) // self.ncut, self.ncut))
    denom = 1 + expval_m.sum(1)
    denom = np.kron(denom, np.ones(self.ncut, dtype=np.float64))
    mprob = expval / denom
    dmat = mprob[:, None] * exog
    ddenom = expval[:, None] * exog
    dmat -= mprob[:, None] * ddenom / denom[:, None]
    return dmat