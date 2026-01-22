from statsmodels.compat.pandas import Appender
import warnings
import numpy as np
from numpy.linalg import LinAlgError
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.base import _prediction_inference as pred
from statsmodels.base._prediction_inference import PredictionResultsMean
import statsmodels.base._parameter_inference as pinfer
from statsmodels.graphics._regressionplots_doc import (
import statsmodels.regression._tools as reg_tools
import statsmodels.regression.linear_model as lm
from statsmodels.tools.decorators import (
from statsmodels.tools.docstring import Docstring
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.validation import float_like
from . import families
def estimate_tweedie_power(self, mu, method='brentq', low=1.01, high=5.0):
    """
        Tweedie specific function to estimate scale and the variance parameter.
        The variance parameter is also referred to as p, xi, or shape.

        Parameters
        ----------
        mu : array_like
            Fitted mean response variable
        method : str, defaults to 'brentq'
            Scipy optimizer used to solve the Pearson equation. Only brentq
            currently supported.
        low : float, optional
            Low end of the bracketing interval [a,b] to be used in the search
            for the power. Defaults to 1.01.
        high : float, optional
            High end of the bracketing interval [a,b] to be used in the search
            for the power. Defaults to 5.

        Returns
        -------
        power : float
            The estimated shape or power.
        """
    if method == 'brentq':
        from scipy.optimize import brentq

        def psi_p(power, mu):
            scale = (self.iweights * (self.endog - mu) ** 2 / mu ** power).sum() / self.df_resid
            return np.sum(self.iweights * ((self.endog - mu) ** 2 / (scale * mu ** power) - 1) * np.log(mu)) / self.freq_weights.sum()
        power = brentq(psi_p, low, high, args=mu)
    else:
        raise NotImplementedError('Only brentq can currently be used')
    return power