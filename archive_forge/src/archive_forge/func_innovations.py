import warnings
import numpy as np
from scipy.optimize import minimize
from statsmodels.tools.tools import Bunch
from statsmodels.tsa.innovations import arma_innovations
from statsmodels.tsa.stattools import acovf, innovations_algo
from statsmodels.tsa.statespace.tools import diff
from statsmodels.tsa.arima.specification import SARIMAXSpecification
from statsmodels.tsa.arima.params import SARIMAXParams
from statsmodels.tsa.arima.estimators.hannan_rissanen import hannan_rissanen
def innovations(endog, ma_order=0, demean=True):
    """
    Estimate MA parameters using innovations algorithm.

    Parameters
    ----------
    endog : array_like or SARIMAXSpecification
        Input time series array, assumed to be stationary.
    ma_order : int, optional
        Maximum moving average order. Default is 0.
    demean : bool, optional
        Whether to estimate and remove the mean from the process prior to
        fitting the moving average coefficients. Default is True.

    Returns
    -------
    parameters : list of SARIMAXParams objects
        List elements correspond to estimates at different `ma_order`. For
        example, parameters[0] is an `SARIMAXParams` instance corresponding to
        `ma_order=0`.
    other_results : Bunch
        Includes one component, `spec`, containing the `SARIMAXSpecification`
        instance corresponding to the input arguments.

    Notes
    -----
    The primary reference is [1]_, section 5.1.3.

    This procedure assumes that the series is stationary.

    References
    ----------
    .. [1] Brockwell, Peter J., and Richard A. Davis. 2016.
       Introduction to Time Series and Forecasting. Springer.
    """
    spec = max_spec = SARIMAXSpecification(endog, ma_order=ma_order)
    endog = max_spec.endog
    if demean:
        endog = endog - endog.mean()
    if not max_spec.is_ma_consecutive:
        raise ValueError('Innovations estimation unavailable for models with seasonal or otherwise non-consecutive MA orders.')
    sample_acovf = acovf(endog, fft=True)
    theta, v = innovations_algo(sample_acovf, nobs=max_spec.ma_order + 1)
    ma_params = [theta[i, :i] for i in range(1, max_spec.ma_order + 1)]
    sigma2 = v
    out = []
    for i in range(max_spec.ma_order + 1):
        spec = SARIMAXSpecification(ma_order=i)
        p = SARIMAXParams(spec=spec)
        if i == 0:
            p.params = sigma2[i]
        else:
            p.params = np.r_[ma_params[i - 1], sigma2[i]]
        out.append(p)
    other_results = Bunch({'spec': spec})
    return (out, other_results)