import numpy
import cupy
from cupy import _core
from cupy._core import internal
from cupyx.scipy.ndimage import _util
from cupyx.scipy import special
def fourier_gaussian(input, sigma, n=-1, axis=-1, output=None):
    """Multidimensional Gaussian shift filter.

    The array is multiplied with the Fourier transform of a (separable)
    Gaussian kernel.

    Args:
        input (cupy.ndarray): The input array.
        sigma (float or sequence of float):  The sigma of the Gaussian kernel.
            If a float, `sigma` is the same for all axes. If a sequence,
            `sigma` has to contain one value for each axis.
        n (int, optional):  If `n` is negative (default), then the input is
            assumed to be the result of a complex fft. If `n` is larger than or
            equal to zero, the input is assumed to be the result of a real fft,
            and `n` gives the length of the array before transformation along
            the real transform direction.
        axis (int, optional): The axis of the real transform (only used when
            ``n > -1``).
        output (cupy.ndarray, optional):
            If given, the result of shifting the input is placed in this array.

    Returns:
        output (cupy.ndarray): The filtered output.
    """
    ndim = input.ndim
    output = _get_output_fourier(output, input)
    axis = internal._normalize_axis_index(axis, ndim)
    sigmas = _util._fix_sequence_arg(sigma, ndim, 'sigma')
    _core.elementwise_copy(input, output)
    for ax, (sigmak, ax_size) in enumerate(zip(sigmas, output.shape)):
        if ax == axis and n > 0:
            arr = cupy.arange(ax_size, dtype=output.real.dtype)
            arr /= n
        else:
            arr = cupy.fft.fftfreq(ax_size)
        arr = arr.astype(output.real.dtype, copy=False)
        arr *= arr
        scale = sigmak * sigmak / -2
        arr *= 4 * numpy.pi * numpy.pi * scale
        cupy.exp(arr, out=arr)
        arr = _reshape_nd(arr, ndim=ndim, axis=ax)
        output *= arr
    return output