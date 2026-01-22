import numpy as np
import scipy.fft as fft
from .._shared.utils import _supported_float_type
def image_quad_norm(inarray):
    """Return the quadratic norm of images in Fourier space.

    This function detects whether the input image satisfies the
    Hermitian property.

    Parameters
    ----------
    inarray : ndarray
        Input image. The image data should reside in the final two
        axes.

    Returns
    -------
    norm : float
        The quadratic norm of ``inarray``.

    Examples
    --------
    >>> input = np.ones((5, 5))
    >>> image_quad_norm(ufft2(input)) == np.sum(np.abs(input)**2)
    True
    >>> image_quad_norm(ufft2(input)) == image_quad_norm(urfft2(input))
    True
    """
    if inarray.shape[-1] != inarray.shape[-2]:
        return 2 * np.sum(np.sum(np.abs(inarray) ** 2, axis=-1), axis=-1) - np.sum(np.abs(inarray[..., 0]) ** 2, axis=-1)
    else:
        return np.sum(np.sum(np.abs(inarray) ** 2, axis=-1), axis=-1)