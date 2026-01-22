from statsmodels.compat.pandas import Appender
from statsmodels.compat.python import lrange, lzip
import numpy as np
import pandas as pd
from patsy import dmatrix
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.graphics import utils
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.regression.linear_model import GLS, OLS, WLS
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.tools.tools import maybe_unwrap_results
from ._regressionplots_doc import (

    Residualize the endog variable and a 'focus' exog variable in a
    regression model with respect to the other exog variables.

    Parameters
    ----------
    results : regression results instance
        A fitted model including the focus exog and all other
        predictors of interest.
    focus_exog : {int, str}
        The column of results.model.exog or a variable name that is
        to be residualized against the other predictors.
    resid_type : str
        The type of residuals to use for the dependent variable.  If
        None, uses `resid_deviance` for GLM/GEE and `resid` otherwise.
    use_glm_weights : bool
        Only used if the model is a GLM or GEE.  If True, the
        residuals for the focus predictor are computed using WLS, with
        the weights obtained from the IRLS calculations for fitting
        the GLM.  If False, unweighted regression is used.
    fit_kwargs : dict, optional
        Keyword arguments to be passed to fit when refitting the
        model.

    Returns
    -------
    endog_resid : array_like
        The residuals for the original exog
    focus_exog_resid : array_like
        The residuals for the focus predictor

    Notes
    -----
    The 'focus variable' residuals are always obtained using linear
    regression.

    Currently only GLM, GEE, and OLS models are supported.
    