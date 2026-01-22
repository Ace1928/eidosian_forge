import warnings
from numbers import Real
import numpy as np
from scipy.special import xlogy
from ..exceptions import UndefinedMetricWarning
from ..utils._param_validation import Hidden, Interval, StrOptions, validate_params
from ..utils.stats import _weighted_percentile
from ..utils.validation import (
@validate_params({'y_true': ['array-like'], 'y_pred': ['array-like'], 'sample_weight': ['array-like', None]}, prefer_skip_nested_validation=True)
def mean_poisson_deviance(y_true, y_pred, *, sample_weight=None):
    """Mean Poisson deviance regression loss.

    Poisson deviance is equivalent to the Tweedie deviance with
    the power parameter `power=1`.

    Read more in the :ref:`User Guide <mean_tweedie_deviance>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values. Requires y_true >= 0.

    y_pred : array-like of shape (n_samples,)
        Estimated target values. Requires y_pred > 0.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    loss : float
        A non-negative floating point value (the best value is 0.0).

    Examples
    --------
    >>> from sklearn.metrics import mean_poisson_deviance
    >>> y_true = [2, 0, 1, 4]
    >>> y_pred = [0.5, 0.5, 2., 2.]
    >>> mean_poisson_deviance(y_true, y_pred)
    1.4260...
    """
    return mean_tweedie_deviance(y_true, y_pred, sample_weight=sample_weight, power=1)