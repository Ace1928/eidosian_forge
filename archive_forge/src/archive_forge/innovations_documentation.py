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

    Estimate SARIMA parameters by MLE using innovations algorithm.

    Parameters
    ----------
    endog : array_like
        Input time series array.
    order : tuple, optional
        The (p,d,q) order of the model for the number of AR parameters,
        differences, and MA parameters. Default is (0, 0, 0).
    seasonal_order : tuple, optional
        The (P,D,Q,s) order of the seasonal component of the model for the
        AR parameters, differences, MA parameters, and periodicity. Default
        is (0, 0, 0, 0).
    demean : bool, optional
        Whether to estimate and remove the mean from the process prior to
        fitting the SARIMA coefficients. Default is True.
    enforce_invertibility : bool, optional
        Whether or not to transform the MA parameters to enforce invertibility
        in the moving average component of the model. Default is True.
    start_params : array_like, optional
        Initial guess of the solution for the loglikelihood maximization. The
        AR polynomial must be stationary. If `enforce_invertibility=True` the
        MA poylnomial must be invertible. If not provided, default starting
        parameters are computed using the Hannan-Rissanen method.
    minimize_kwargs : dict, optional
        Arguments to pass to scipy.optimize.minimize.

    Returns
    -------
    parameters : SARIMAXParams object
    other_results : Bunch
        Includes four components: `spec`, containing the `SARIMAXSpecification`
        instance corresponding to the input arguments; `minimize_kwargs`,
        containing any keyword arguments passed to `minimize`; `start_params`,
        containing the untransformed starting parameters passed to `minimize`;
        and `minimize_results`, containing the output from `minimize`.

    Notes
    -----
    The primary reference is [1]_, section 5.2.

    Note: we do not include `enforce_stationarity` as an argument, because this
    function requires stationarity.

    TODO: support concentrating out the scale (should be easy: use sigma2=1
          and then compute sigma2=np.sum(u**2 / v) / len(u); would then need to
          redo llf computation in the Cython function).

    TODO: add support for fixed parameters

    TODO: add support for secondary optimization that does not enforce
          stationarity / invertibility, starting from first step's parameters

    References
    ----------
    .. [1] Brockwell, Peter J., and Richard A. Davis. 2016.
       Introduction to Time Series and Forecasting. Springer.
    