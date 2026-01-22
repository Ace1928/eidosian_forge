import numpy as np
import warnings
from statsmodels.tools.tools import add_constant, Bunch
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.innovations import arma_innovations
from statsmodels.tsa.statespace.tools import diff
from statsmodels.tsa.arima.estimators.yule_walker import yule_walker
from statsmodels.tsa.arima.estimators.burg import burg
from statsmodels.tsa.arima.estimators.hannan_rissanen import hannan_rissanen
from statsmodels.tsa.arima.estimators.innovations import (
from statsmodels.tsa.arima.estimators.statespace import statespace
from statsmodels.tsa.arima.specification import SARIMAXSpecification
from statsmodels.tsa.arima.params import SARIMAXParams

    Estimate ARMAX parameters by GLS.

    Parameters
    ----------
    endog : array_like
        Input time series array.
    exog : array_like, optional
        Array of exogenous regressors. If not included, then `include_constant`
        must be True, and then `exog` will only include the constant column.
    order : tuple, optional
        The (p,d,q) order of the ARIMA model. Default is (0, 0, 0).
    seasonal_order : tuple, optional
        The (P,D,Q,s) order of the seasonal ARIMA model.
        Default is (0, 0, 0, 0).
    include_constant : bool, optional
        Whether to add a constant term in `exog` if it's not already there.
        The estimate of the constant will then appear as one of the `exog`
        parameters. If `exog` is None, then the constant will represent the
        mean of the process. Default is True if the specified model does not
        include integration and False otherwise.
    n_iter : int, optional
        Optionally iterate feasible GSL a specific number of times. Default is
        to iterate to convergence. If set, this argument overrides the
        `max_iter` and `tolerance` arguments.
    max_iter : int, optional
        Maximum number of feasible GLS iterations. Default is 50. If `n_iter`
        is set, it overrides this argument.
    tolerance : float, optional
        Tolerance for determining convergence of feasible GSL iterations. If
        `iter` is set, this argument has no effect.
        Default is 1e-8.
    arma_estimator : str, optional
        The estimator used for estimating the ARMA model. This option should
        not generally be used, unless the default method is failing or is
        otherwise unsuitable. Not all values will be valid, depending on the
        specified model orders (`order` and `seasonal_order`). Possible values
        are:
        * 'innovations_mle' - can be used with any specification
        * 'statespace' - can be used with any specification
        * 'hannan_rissanen' - can be used with any ARMA non-seasonal model
        * 'yule_walker' - only non-seasonal consecutive
          autoregressive (AR) models
        * 'burg' - only non-seasonal, consecutive autoregressive (AR) models
        * 'innovations' - only non-seasonal, consecutive moving
          average (MA) models.
        The default is 'innovations_mle'.
    arma_estimator_kwargs : dict, optional
        Arguments to pass to the ARMA estimator.

    Returns
    -------
    parameters : SARIMAXParams object
        Contains the parameter estimates from the final iteration.
    other_results : Bunch
        Includes eight components: `spec`, `params`, `converged`,
        `differences`, `iterations`, `arma_estimator`, 'arma_estimator_kwargs',
        and `arma_results`.

    Notes
    -----
    The primary reference is [1]_, section 6.6. In particular, the
    implementation follows the iterative procedure described in section 6.6.2.
    Construction of the transformed variables used to compute the GLS estimator
    described in section 6.6.1 is done via an application of the innovations
    algorithm (rather than explicit construction of the transformation matrix).

    Note that if the specified model includes integration, both the `endog` and
    `exog` series will be differenced prior to estimation and a warning will
    be issued to alert the user.

    References
    ----------
    .. [1] Brockwell, Peter J., and Richard A. Davis. 2016.
       Introduction to Time Series and Forecasting. Springer.
    