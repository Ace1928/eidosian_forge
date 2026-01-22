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
def select_penweight_kfold(self, alphas=None, cv_iterator=None, cost=None, k_folds=5, k_grid=11):
    """find alphas by k-fold cross-validation

        Warning: This estimates ``k_folds`` models for each point in the
            grid of alphas.

        Parameters
        ----------
        alphas : None or list of arrays
        cv_iterator : instance
            instance of a cross-validation iterator, by default this is a
            KFold instance
        cost : function
            default is mean squared error. The cost function to evaluate the
            prediction error for the left out sample. This should take two
            arrays as argument and return one float.
        k_folds : int
            number of folds if default Kfold iterator is used.
            This is ignored if ``cv_iterator`` is not None.

        Returns
        -------
        alpha_cv : list of float
            Best alpha in grid according to cross-validation
        res_cv : instance of MultivariateGAMCVPath
            The instance was used for cross-validation and holds the results

        Notes
        -----
        The default alphas are defined as
        ``alphas = [np.logspace(0, 7, k_grid) for _ in range(k_smooths)]``
        """
    if cost is None:

        def cost(x1, x2):
            return np.linalg.norm(x1 - x2) / len(x1)
    if alphas is None:
        alphas = [np.logspace(0, 7, k_grid) for _ in range(self.k_smooths)]
    if cv_iterator is None:
        cv_iterator = KFold(k_folds=k_folds, shuffle=True)
    gam_cv = MultivariateGAMCVPath(smoother=self.smoother, alphas=alphas, gam=GLMGam, cost=cost, endog=self.endog, exog=self.exog_linear, cv_iterator=cv_iterator)
    gam_cv_res = gam_cv.fit()
    return (gam_cv_res.alpha_cv, gam_cv_res)