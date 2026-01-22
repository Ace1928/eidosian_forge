import numbers
import numpy as np
def scale_transform(data, center='median', transform='abs', trim_frac=0.2, axis=0):
    """Transform data for variance comparison for Levene type tests

    Parameters
    ----------
    data : array_like
        Observations for the data.
    center : "median", "mean", "trimmed" or float
        Statistic used for centering observations. If a float, then this
        value is used to center. Default is median.
    transform : 'abs', 'square', 'identity' or a callable
        The transform for the centered data.
    trim_frac : float in [0, 0.5)
        Fraction of observations that are trimmed on each side of the sorted
        observations. This is only used if center is `trimmed`.
    axis : int
        Axis along which the data are transformed when centering.

    Returns
    -------
    res : ndarray
        transformed data in the same shape as the original data.

    """
    x = np.asarray(data)
    if transform == 'abs':
        tfunc = np.abs
    elif transform == 'square':
        tfunc = lambda x: x * x
    elif transform == 'identity':
        tfunc = lambda x: x
    elif callable(transform):
        tfunc = transform
    else:
        raise ValueError('transform should be abs, square or exp')
    if center == 'median':
        res = tfunc(x - np.expand_dims(np.median(x, axis=axis), axis))
    elif center == 'mean':
        res = tfunc(x - np.expand_dims(np.mean(x, axis=axis), axis))
    elif center == 'trimmed':
        center = trim_mean(x, trim_frac, axis=axis)
        res = tfunc(x - np.expand_dims(center, axis))
    elif isinstance(center, numbers.Number):
        res = tfunc(x - center)
    else:
        raise ValueError('center should be median, mean or trimmed')
    return res