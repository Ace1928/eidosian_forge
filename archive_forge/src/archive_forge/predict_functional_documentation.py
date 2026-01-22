import pandas as pd
import patsy
import numpy as np
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.compat.pandas import Appender

    The basic SCR from (Sun et al. Annals of Statistics 2000).

    Computes simultaneous confidence regions (SCR).

    Parameters
    ----------
    result : results instance
        The fitted GLM results instance
    exog : array_like
        The exog values spanning the interval
    alpha : float
        `1 - alpha` is the coverage probability.

    Returns
    -------
    An array with two columns, containing the lower and upper
    confidence bounds, respectively.

    Notes
    -----
    The rows of `exog` should be a sequence of covariate values
    obtained by taking one 'free variable' x and varying it over an
    interval.  The matrix `exog` is thus the basis functions and any
    other covariates evaluated as x varies.
    