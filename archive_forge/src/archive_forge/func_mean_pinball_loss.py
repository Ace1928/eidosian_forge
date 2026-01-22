import warnings
from numbers import Real
import numpy as np
from scipy.special import xlogy
from ..exceptions import UndefinedMetricWarning
from ..utils._param_validation import Hidden, Interval, StrOptions, validate_params
from ..utils.stats import _weighted_percentile
from ..utils.validation import (
@validate_params({'y_true': ['array-like'], 'y_pred': ['array-like'], 'sample_weight': ['array-like', None], 'alpha': [Interval(Real, 0, 1, closed='both')], 'multioutput': [StrOptions({'raw_values', 'uniform_average'}), 'array-like']}, prefer_skip_nested_validation=True)
def mean_pinball_loss(y_true, y_pred, *, sample_weight=None, alpha=0.5, multioutput='uniform_average'):
    """Pinball loss for quantile regression.

    Read more in the :ref:`User Guide <pinball_loss>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    alpha : float, slope of the pinball loss, default=0.5,
        This loss is equivalent to :ref:`mean_absolute_error` when `alpha=0.5`,
        `alpha=0.95` is minimized by estimators of the 95th percentile.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape             (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.

        'raw_values' :
            Returns a full set of errors in case of multioutput input.

        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

    Returns
    -------
    loss : float or ndarray of floats
        If multioutput is 'raw_values', then mean absolute error is returned
        for each output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average of all output errors is returned.

        The pinball loss output is a non-negative floating point. The best
        value is 0.0.

    Examples
    --------
    >>> from sklearn.metrics import mean_pinball_loss
    >>> y_true = [1, 2, 3]
    >>> mean_pinball_loss(y_true, [0, 2, 3], alpha=0.1)
    0.03...
    >>> mean_pinball_loss(y_true, [1, 2, 4], alpha=0.1)
    0.3...
    >>> mean_pinball_loss(y_true, [0, 2, 3], alpha=0.9)
    0.3...
    >>> mean_pinball_loss(y_true, [1, 2, 4], alpha=0.9)
    0.03...
    >>> mean_pinball_loss(y_true, y_true, alpha=0.1)
    0.0
    >>> mean_pinball_loss(y_true, y_true, alpha=0.9)
    0.0
    """
    y_type, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    check_consistent_length(y_true, y_pred, sample_weight)
    diff = y_true - y_pred
    sign = (diff >= 0).astype(diff.dtype)
    loss = alpha * sign * diff - (1 - alpha) * (1 - sign) * diff
    output_errors = np.average(loss, weights=sample_weight, axis=0)
    if isinstance(multioutput, str) and multioutput == 'raw_values':
        return output_errors
    if isinstance(multioutput, str) and multioutput == 'uniform_average':
        multioutput = None
    return np.average(output_errors, weights=multioutput)