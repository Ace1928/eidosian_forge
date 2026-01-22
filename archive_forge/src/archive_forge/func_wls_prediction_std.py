from __future__ import annotations
import warnings
from contextlib import suppress
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from .._utils import get_valid_kwargs
from ..exceptions import PlotnineError, PlotnineWarning
def wls_prediction_std(res, exog=None, weights=None, alpha=0.05, interval='confidence'):
    """
    Calculate standard deviation and confidence interval

    Applies to WLS and OLS, not to general GLS,
    that is independently but not identically distributed observations

    Parameters
    ----------
    res : regression-result
        results of WLS or OLS regression required attributes see notes
    exog : array_like
        exogenous variables for points to predict
    weights : scalar | array_like
        weights as defined for WLS (inverse of variance of observation)
    alpha : float
        confidence level for two-sided hypothesis
    interval : str
        Type of interval to compute. One of "confidence" or "prediction"

    Returns
    -------
    predstd : array_like
        Standard error of prediction. It must be the same length as rows
        of exog.
    interval_l, interval_u : array_like
        Lower und upper confidence bounds

    Notes
    -----
    The result instance needs to have at least the following
    res.model.predict() : predicted values or
    res.fittedvalues : values used in estimation
    res.cov_params() : covariance matrix of parameter estimates

    If exog is 1d, then it is interpreted as one observation,
    i.e. a row vector.

    testing status: not compared with other packages

    References
    ----------
    Greene p.111 for OLS, extended to WLS by analogy
    """
    import scipy.stats as stats
    covb = res.cov_params()
    if exog is None:
        exog = res.model.exog
        predicted = res.fittedvalues
        if weights is None:
            weights = res.model.weights
    else:
        exog = np.atleast_2d(exog)
        if covb.shape[1] != exog.shape[1]:
            raise ValueError('wrong shape of exog')
        predicted = res.model.predict(res.params, exog)
        if weights is None:
            weights = 1.0
        else:
            weights = np.asarray(weights)
            if weights.size > 1 and len(weights) != exog.shape[0]:
                raise ValueError('weights and exog do not have matching shape')
    predvar = res.mse_resid / weights
    ip = (exog * np.dot(covb, exog.T).T).sum(1)
    if interval == 'confidence':
        predstd = np.sqrt(ip)
    elif interval == 'prediction':
        predstd = np.sqrt(ip + predvar)
    else:
        raise ValueError(f'Unknown value for interval={interval!r}')
    tppf = stats.t.isf(alpha / 2.0, res.df_resid)
    interval_u = predicted + tppf * predstd
    interval_l = predicted - tppf * predstd
    return (predstd, interval_l, interval_u)