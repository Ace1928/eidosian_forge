import numpy as np
import scipy.fft as fft
from .._shared.utils import _supported_float_type
def uifft2(inarray):
    """2-dimensional inverse unitary Fourier transform.

    Compute the inverse Fourier transform on the last 2 axes.

    Parameters
    ----------
    inarray : ndarray
        The array to transform.

    Returns
    -------
    outarray : ndarray (same shape as inarray)
        The unitary 2-D inverse Fourier transform of ``inarray``.

    See Also
    --------
    uifft2, uifftn, uirfftn

    Examples
    --------
    >>> input = np.ones((10, 128, 128))
    >>> output = uifft2(input)
    >>> np.allclose(np.sum(input[1, ...]) / np.sqrt(input[1, ...].size),
    ...             output[0, 0, 0])
    True
    >>> output.shape
    (10, 128, 128)
    """
    return uifftn(inarray, 2)