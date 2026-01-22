import warnings
from numbers import Real
import numpy as np
from scipy.special import xlogy
from ..exceptions import UndefinedMetricWarning
from ..utils._param_validation import Hidden, Interval, StrOptions, validate_params
from ..utils.stats import _weighted_percentile
from ..utils.validation import (
@validate_params({'y_true': ['array-like'], 'y_pred': ['array-like'], 'multioutput': [StrOptions({'raw_values', 'uniform_average'}), 'array-like'], 'sample_weight': ['array-like', None]}, prefer_skip_nested_validation=True)
def median_absolute_error(y_true, y_pred, *, multioutput='uniform_average', sample_weight=None):
    """Median absolute error regression loss.

    Median absolute error output is non-negative floating point. The best value
    is 0.0. Read more in the :ref:`User Guide <median_absolute_error>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    multioutput : {'raw_values', 'uniform_average'} or array-like of shape             (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values. Array-like value defines
        weights used to average errors.

        'raw_values' :
            Returns a full set of errors in case of multioutput input.

        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

        .. versionadded:: 0.24

    Returns
    -------
    loss : float or ndarray of floats
        If multioutput is 'raw_values', then mean absolute error is returned
        for each output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average of all output errors is returned.

    Examples
    --------
    >>> from sklearn.metrics import median_absolute_error
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> median_absolute_error(y_true, y_pred)
    0.5
    >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
    >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
    >>> median_absolute_error(y_true, y_pred)
    0.75
    >>> median_absolute_error(y_true, y_pred, multioutput='raw_values')
    array([0.5, 1. ])
    >>> median_absolute_error(y_true, y_pred, multioutput=[0.3, 0.7])
    0.85
    """
    y_type, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    if sample_weight is None:
        output_errors = np.median(np.abs(y_pred - y_true), axis=0)
    else:
        sample_weight = _check_sample_weight(sample_weight, y_pred)
        output_errors = _weighted_percentile(np.abs(y_pred - y_true), sample_weight=sample_weight)
    if isinstance(multioutput, str):
        if multioutput == 'raw_values':
            return output_errors
        elif multioutput == 'uniform_average':
            multioutput = None
    return np.average(output_errors, weights=multioutput)