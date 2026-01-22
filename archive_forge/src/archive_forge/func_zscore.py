import cupy
def zscore(a, axis=0, ddof=0, nan_policy='propagate'):
    """Compute the z-score.

    Compute the z-score of each value in the sample, relative to
    the sample mean and standard deviation.

    Parameters
    ----------
    a : array-like
        An array like object containing the sample data
    axis : int or None, optional
        Axis along which to operate. Default is 0. If None,
        compute over the whole arrsy `a`
    ddof : int, optional
        Degrees of freedom correction in the calculation of the
        standard deviation. Default is 0
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan. 'propagate'
        returns nan, 'raise' throws an error, 'omit' performs
        the calculations ignoring nan values. Default is
        'propagate'. Note that when the value is 'omit',
        nans in the input also propagate to the output,
        but they do not affect the z-scores computed
        for the non-nan values

    Returns
    -------
    zscore : array-like
        The z-scores, standardized by mean and standard deviation of
        input array `a`

    """
    return zmap(a, a, axis=axis, ddof=ddof, nan_policy=nan_policy)