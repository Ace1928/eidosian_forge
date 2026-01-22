import warnings
import numpy as np
from numpy.linalg import eigh, inv, norm, matrix_rank
import pandas as pd
from scipy.optimize import minimize
from statsmodels.tools.decorators import cache_readonly
from statsmodels.base.model import Model
from statsmodels.iolib import summary2
from statsmodels.graphics.utils import _import_mpl
from .factor_rotation import rotate_factors, promax
@cache_readonly
def uniq_stderr(self, kurt=0):
    """
        The standard errors of the uniquenesses.

        Parameters
        ----------
        kurt : float
            Excess kurtosis

        Notes
        -----
        If excess kurtosis is known, provide as `kurt`.  Standard
        errors are only available if the model was fit using maximum
        likelihood.  If `endog` is not provided, `nobs` must be
        provided to obtain standard errors.

        These are asymptotic standard errors.  See Bai and Li (2012)
        for conditions under which the standard errors are valid.

        The standard errors are only applicable to the original,
        unrotated maximum likelihood solution.
        """
    if self.fa_method.lower() != 'ml':
        msg = 'Standard errors only available under ML estimation'
        raise ValueError(msg)
    if self.nobs is None:
        msg = 'nobs is required to obtain standard errors.'
        raise ValueError(msg)
    v = self.uniqueness ** 2 * (2 + kurt)
    return np.sqrt(v / self.nobs)