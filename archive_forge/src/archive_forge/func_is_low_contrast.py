import numpy as np
from ..util.dtype import dtype_range, dtype_limits
from .._shared import utils
def is_low_contrast(image, fraction_threshold=0.05, lower_percentile=1, upper_percentile=99, method='linear'):
    """Determine if an image is low contrast.

    Parameters
    ----------
    image : array-like
        The image under test.
    fraction_threshold : float, optional
        The low contrast fraction threshold. An image is considered low-
        contrast when its range of brightness spans less than this
        fraction of its data type's full range. [1]_
    lower_percentile : float, optional
        Disregard values below this percentile when computing image contrast.
    upper_percentile : float, optional
        Disregard values above this percentile when computing image contrast.
    method : str, optional
        The contrast determination method.  Right now the only available
        option is "linear".

    Returns
    -------
    out : bool
        True when the image is determined to be low contrast.

    Notes
    -----
    For boolean images, this function returns False only if all values are
    the same (the method, threshold, and percentile arguments are ignored).

    References
    ----------
    .. [1] https://scikit-image.org/docs/dev/user_guide/data_types.html

    Examples
    --------
    >>> image = np.linspace(0, 0.04, 100)
    >>> is_low_contrast(image)
    True
    >>> image[-1] = 1
    >>> is_low_contrast(image)
    True
    >>> is_low_contrast(image, upper_percentile=100)
    False
    """
    image = np.asanyarray(image)
    if image.dtype == bool:
        return not (image.max() == 1 and image.min() == 0)
    if image.ndim == 3:
        from ..color import rgb2gray, rgba2rgb
        if image.shape[2] == 4:
            image = rgba2rgb(image)
        if image.shape[2] == 3:
            image = rgb2gray(image)
    dlimits = dtype_limits(image, clip_negative=False)
    limits = np.percentile(image, [lower_percentile, upper_percentile])
    ratio = (limits[1] - limits[0]) / (dlimits[1] - dlimits[0])
    return ratio < fraction_threshold