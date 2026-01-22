import numpy as np
import scipy.fft as fft
from .._shared.utils import _supported_float_type
def urfft2(inarray):
    """2-dimensional real unitary Fourier transform

    Compute the real Fourier transform on the last 2 axes. This
    transform considers the Hermitian property of the transform from
    complex to real-valued input.

    Parameters
    ----------
    inarray : ndarray, shape (M[, ...], P)
        The array to transform.

    Returns
    -------
    outarray : ndarray, shape (M[, ...], 2 * (P - 1))
        The unitary 2-D real Fourier transform of ``inarray``.

    See Also
    --------
    ufft2, ufftn, urfftn

    Examples
    --------
    >>> input = np.ones((10, 128, 128))
    >>> output = urfft2(input)
    >>> np.allclose(np.sum(input[1,...]) / np.sqrt(input[1,...].size),
    ...             output[1, 0, 0])
    True
    >>> output.shape
    (10, 128, 65)
    """
    return urfftn(inarray, 2)