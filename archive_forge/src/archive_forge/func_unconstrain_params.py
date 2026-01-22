import numpy as np
import pandas as pd
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.base.tsa_model import TimeSeriesModel
from statsmodels.tsa.statespace.tools import (
from statsmodels.tsa.arima.tools import standardize_lag_order, validate_basic
def unconstrain_params(self, constrained):
    """
        Reverse transformations used to constrain parameter values to be valid.

        Parameters
        ----------
        constrained : array_like
            Array of model parameters.

        Returns
        -------
        unconstrained : ndarray
            Array of parameters with constraining transformions reversed.

        Notes
        -----
        This is usually only used when performing numerical minimization
        of the log-likelihood function. This function is the (approximate)
        inverse of `constrain_params`.

        Examples
        --------
        >>> spec = SARIMAXSpecification(ar_order=1)
        >>> spec.unconstrain_params([-0.5, 4.])
        array([0.57735, 2.     ])
        """
    constrained = self.split_params(constrained)
    params = {}
    if self.k_exog_params:
        params['exog_params'] = constrained['exog_params']
    if self.k_ar_params:
        if self.enforce_stationarity:
            params['ar_params'] = unconstrain(constrained['ar_params'])
        else:
            params['ar_params'] = constrained['ar_params']
    if self.k_ma_params:
        if self.enforce_invertibility:
            params['ma_params'] = unconstrain(-constrained['ma_params'])
        else:
            params['ma_params'] = constrained['ma_params']
    if self.k_seasonal_ar_params:
        if self.enforce_stationarity:
            params['seasonal_ar_params'] = unconstrain(constrained['seasonal_ar_params'])
        else:
            params['seasonal_ar_params'] = constrained['seasonal_ar_params']
    if self.k_seasonal_ma_params:
        if self.enforce_invertibility:
            params['seasonal_ma_params'] = unconstrain(-constrained['seasonal_ma_params'])
        else:
            params['seasonal_ma_params'] = constrained['seasonal_ma_params']
    if not self.concentrate_scale:
        params['sigma2'] = constrained['sigma2'] ** 0.5
    return self.join_params(**params)