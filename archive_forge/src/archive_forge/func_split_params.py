import numpy as np
import pandas as pd
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.base.tsa_model import TimeSeriesModel
from statsmodels.tsa.statespace.tools import (
from statsmodels.tsa.arima.tools import standardize_lag_order, validate_basic
def split_params(self, params, allow_infnan=False):
    """
        Split parameter array by type into dictionary.

        Parameters
        ----------
        params : array_like
            Array of model parameters.
        allow_infnan : bool, optional
            Whether or not to allow `params` to contain -np.inf, np.inf, and
            np.nan. Default is False.

        Returns
        -------
        split_params : dict
            Dictionary with keys 'exog_params', 'ar_params', 'ma_params',
            'seasonal_ar_params', 'seasonal_ma_params', and (unless
            `concentrate_scale=True`) 'sigma2'. Values are the parameters
            associated with the key, based on the `params` argument.

        Examples
        --------
        >>> spec = SARIMAXSpecification(ar_order=1)
        >>> spec.split_params([0.5, 4])
        {'exog_params': array([], dtype=float64),
         'ar_params': array([0.5]),
         'ma_params': array([], dtype=float64),
         'seasonal_ar_params': array([], dtype=float64),
         'seasonal_ma_params': array([], dtype=float64),
         'sigma2': 4.0}
        """
    params = validate_basic(params, self.k_params, allow_infnan=allow_infnan, title='joint parameters')
    ix = [self.k_exog_params, self.k_ar_params, self.k_ma_params, self.k_seasonal_ar_params, self.k_seasonal_ma_params]
    names = ['exog_params', 'ar_params', 'ma_params', 'seasonal_ar_params', 'seasonal_ma_params']
    if not self.concentrate_scale:
        ix.append(1)
        names.append('sigma2')
    ix = np.cumsum(ix)
    out = dict(zip(names, np.split(params, ix)))
    if 'sigma2' in out:
        out['sigma2'] = out['sigma2'].item()
    return out