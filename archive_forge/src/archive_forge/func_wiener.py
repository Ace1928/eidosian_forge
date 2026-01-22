import warnings
import cupy
from cupy._core import internal
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters
from cupyx.scipy.signal import _signaltools_core as _st_core
def wiener(im, mysize=None, noise=None):
    """Perform a Wiener filter on an N-dimensional array.

    Apply a Wiener filter to the N-dimensional array `im`.

    Args:
        im (cupy.ndarray): An N-dimensional array.
        mysize (int or cupy.ndarray, optional): A scalar or an N-length list
            giving the size of the Wiener filter window in each dimension.
            Elements of mysize should be odd. If mysize is a scalar, then this
            scalar is used as the size in each dimension.
        noise (float, optional): The noise-power to use. If None, then noise is
            estimated as the average of the local variance of the input.

    Returns:
        cupy.ndarray: Wiener filtered result with the same shape as `im`.

    .. seealso:: :func:`scipy.signal.wiener`
    """
    if mysize is None:
        mysize = 3
    mysize = _util._fix_sequence_arg(mysize, im.ndim, 'mysize', int)
    im = im.astype(cupy.complex128 if im.dtype.kind == 'c' else cupy.float64, copy=False)
    local_mean = _filters.uniform_filter(im, mysize, mode='constant')
    local_var = _filters.uniform_filter(im * im, mysize, mode='constant')
    local_var -= local_mean * local_mean
    if noise is None:
        noise = local_var.mean()
    res = im - local_mean
    res *= 1 - noise / local_var
    res += local_mean
    return cupy.where(local_var < noise, local_mean, res)