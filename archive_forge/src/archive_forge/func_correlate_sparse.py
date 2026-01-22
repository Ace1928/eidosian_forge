import numpy as np
from .._shared.utils import _supported_float_type, _to_np_mode
def correlate_sparse(image, kernel, mode='reflect'):
    """Compute valid cross-correlation of `padded_array` and `kernel`.

    This function is *fast* when `kernel` is large with many zeros.

    See ``scipy.ndimage.correlate`` for a description of cross-correlation.

    Parameters
    ----------
    image : ndarray, dtype float, shape (M, N[, ...], P)
        The input array. If mode is 'valid', this array should already be
        padded, as a margin of the same shape as kernel will be stripped
        off.
    kernel : ndarray, dtype float, shape (Q, R[, ...], S)
        The kernel to be correlated. Must have the same number of
        dimensions as `padded_array`. For high performance, it should
        be sparse (few nonzero entries).
    mode : string, optional
        See `scipy.ndimage.correlate` for valid modes.
        Additionally, mode 'valid' is accepted, in which case no padding is
        applied and the result is the result for the smaller image for which
        the kernel is entirely inside the original data.

    Returns
    -------
    result : array of float, shape (M, N[, ...], P)
        The result of cross-correlating `image` with `kernel`. If mode
        'valid' is used, the resulting shape is (M-Q+1, N-R+1[, ...], P-S+1).
    """
    kernel = np.asarray(kernel)
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)
    if mode == 'valid':
        padded_image = image
    else:
        np_mode = _to_np_mode(mode)
        _validate_window_size(kernel.shape)
        padded_image = np.pad(image, [(w // 2, w // 2) for w in kernel.shape], mode=np_mode)
    indices = np.nonzero(kernel)
    values = list(kernel[indices].astype(float_dtype, copy=False))
    indices = list(zip(*indices))
    corner_index = (0,) * kernel.ndim
    if corner_index not in indices:
        indices = [corner_index] + indices
        values = [0.0] + values
    return _correlate_sparse(padded_image, kernel.shape, indices, values)