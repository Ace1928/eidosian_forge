import numpy as np
import scipy.fft as fft
from .._shared.utils import _supported_float_type
def uirfft2(inarray, shape=None):
    """2-dimensional inverse real unitary Fourier transform.

    Compute the real inverse Fourier transform on the last 2 axes.
    This transform considers the Hermitian property of the transform
    from complex to real-valued input.

    Parameters
    ----------
    inarray : ndarray, shape (M[, ...], P)
        The array to transform.
    shape : tuple of int, optional
        The shape of the output. The shape of ``rfft`` is ambiguous in
        case of odd-valued input shape. In this case, this parameter
        should be provided. See ``np.fft.irfftn``.

    Returns
    -------
    outarray : ndarray, shape (M[, ...], 2 * (P - 1))
        The unitary 2-D inverse real Fourier transform of ``inarray``.

    See Also
    --------
    urfft2, uifftn, uirfftn

    Examples
    --------
    >>> input = np.ones((10, 128, 128))
    >>> output = uirfftn(urfftn(input), shape=input.shape)
    >>> np.allclose(input, output)
    True
    >>> output.shape
    (10, 128, 128)
    """
    return uirfftn(inarray, 2, shape=shape)