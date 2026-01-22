from statsmodels.compat.numpy import NP_LT_2
from statsmodels.compat.pandas import Appender
import warnings
import numpy as np
from scipy import linalg, optimize, signal
from statsmodels.tools.docstring import Docstring, remove_parameters
from statsmodels.tools.validation import array_like
@classmethod
def from_estimation(cls, model_results, nobs=None):
    """
        Create an ArmaProcess from the results of an ARIMA estimation.

        Parameters
        ----------
        model_results : ARIMAResults instance
            A fitted model.
        nobs : int, optional
            If None, nobs is taken from the results.

        Returns
        -------
        ArmaProcess
            Class instance initialized from model_results.

        See Also
        --------
        statsmodels.tsa.arima.model.ARIMA
            The models class used to create the ArmaProcess
        """
    nobs = nobs or model_results.nobs
    return cls(model_results.polynomial_reduced_ar, model_results.polynomial_reduced_ma, nobs=nobs)