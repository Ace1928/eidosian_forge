import warnings
import cupy
from cupy._core import internal
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters
from cupyx.scipy.signal import _signaltools_core as _st_core
def choose_conv_method(in1, in2, mode='full'):
    """Find the fastest convolution/correlation method.

    Args:
        in1 (cupy.ndarray): first input.
        in2 (cupy.ndarray): second input.
        mode (str, optional): ``'valid'``, ``'same'``, ``'full'``.

    Returns:
        str: A string indicating which convolution method is fastest,
        either ``'direct'`` or ``'fft'``.

    .. warning::
        This function currently doesn't support measure option,
        nor multidimensional inputs. It does not guarantee
        the compatibility of the return value to SciPy's one.

    .. seealso:: :func:`scipy.signal.choose_conv_method`

    """
    return cupy._math.misc._choose_conv_method(in1, in2, mode)