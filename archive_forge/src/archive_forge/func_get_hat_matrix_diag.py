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
def get_hat_matrix_diag(self, observed=True, _axis=1):
    """
        Compute the diagonal of the hat matrix

        Parameters
        ----------
        observed : bool
            If true, then observed hessian is used in the hat matrix
            computation. If false, then the expected hessian is used.
            In the case of a canonical link function both are the same.
            This is only relevant for models that implement both observed
            and expected Hessian, which is currently only GLM. Other
            models only use the observed Hessian.
        _axis : int
            This is mainly for internal use. By default it returns the usual
            diagonal of the hat matrix. If _axis is zero, then the result
            corresponds to the effective degrees of freedom, ``edf`` for each
            column of exog.

        Returns
        -------
        hat_matrix_diag : ndarray
            The diagonal of the hat matrix computed from the observed
            or expected hessian.
        """
    weights = self.model.hessian_factor(self.params, scale=self.scale, observed=observed)
    wexog = np.sqrt(weights)[:, None] * self.model.exog
    hess_inv = self.normalized_cov_params * self.scale
    hd = (wexog * hess_inv.dot(wexog.T).T).sum(axis=_axis)
    return hd