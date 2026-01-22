import numpy as np
def validate_basic(params, length, allow_infnan=False, title=None):
    """
    Validate parameter vector for basic correctness.

    Parameters
    ----------
    params : array_like
        Array of parameters to validate.
    length : int
        Expected length of the parameter vector.
    allow_infnan : bool, optional
            Whether or not to allow `params` to contain -np.inf, np.inf, and
            np.nan. Default is False.
    title : str, optional
        Description of the parameters (e.g. "autoregressive") to use in error
        messages.

    Returns
    -------
    params : ndarray
        Array of validated parameters.

    Notes
    -----
    Basic check that the parameters are numeric and that they are the right
    shape. Optionally checks for NaN / infinite values.
    """
    title = '' if title is None else ' for %s' % title
    try:
        params = np.array(params, dtype=object)
        is_complex = [isinstance(p, complex) for p in params.ravel()]
        dtype = complex if any(is_complex) else float
        params = np.array(params, dtype=dtype)
    except TypeError:
        raise ValueError('Parameters vector%s includes invalid values.' % title)
    if not allow_infnan and (np.any(np.isnan(params)) or np.any(np.isinf(params))):
        raise ValueError('Parameters vector%s includes NaN or Inf values.' % title)
    params = np.atleast_1d(np.squeeze(params))
    if params.shape != (length,):
        plural = '' if length == 1 else 's'
        raise ValueError('Specification%s implies %d parameter%s, but values with shape %s were provided.' % (title, length, plural, params.shape))
    return params