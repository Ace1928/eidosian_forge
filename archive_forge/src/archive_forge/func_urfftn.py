import numpy as np
import scipy.fft as fft
from .._shared.utils import _supported_float_type
def urfftn(inarray, dim=None):
    """N-dimensional real unitary Fourier transform.

    This transform considers the Hermitian property of the transform on
    real-valued input.

    Parameters
    ----------
    inarray : ndarray, shape (M[, ...], P)
        The array to transform.
    dim : int, optional
        The last axis along which to compute the transform. All
        axes by default.

    Returns
    -------
    outarray : ndarray, shape (M[, ...], P / 2 + 1)
        The unitary N-D real Fourier transform of ``inarray``.

    Notes
    -----
    The ``urfft`` functions assume an input array of real
    values. Consequently, the output has a Hermitian property and
    redundant values are not computed or returned.

    Examples
    --------
    >>> input = np.ones((5, 5, 5))
    >>> output = urfftn(input)
    >>> np.allclose(np.sum(input) / np.sqrt(input.size), output[0, 0, 0])
    True
    >>> output.shape
    (5, 5, 3)
    """
    if dim is None:
        dim = inarray.ndim
    outarray = fft.rfftn(inarray, axes=range(-dim, 0), norm='ortho')
    return outarray