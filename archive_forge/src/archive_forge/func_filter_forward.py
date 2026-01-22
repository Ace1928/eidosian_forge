import numpy as np
import scipy.fft as fft
from .._shared.utils import _supported_float_type, check_nD
def filter_forward(data, impulse_response=None, filter_params=None, predefined_filter=None):
    """Apply the given filter to data.

    Parameters
    ----------
    data : (M, N) ndarray
        Input data.
    impulse_response : callable `f(r, c, **filter_params)`
        Impulse response of the filter.  See LPIFilter2D.__init__.
    filter_params : dict, optional
        Additional keyword parameters to the impulse_response function.

    Other Parameters
    ----------------
    predefined_filter : LPIFilter2D
        If you need to apply the same filter multiple times over different
        images, construct the LPIFilter2D and specify it here.

    Examples
    --------

    Gaussian filter without normalization:

    >>> def filt_func(r, c, sigma=1):
    ...     return np.exp(-(r**2 + c**2)/(2 * sigma**2))
    >>>
    >>> from skimage import data
    >>> filtered = filter_forward(data.coins(), filt_func)

    """
    if filter_params is None:
        filter_params = {}
    check_nD(data, 2, 'data')
    if predefined_filter is None:
        predefined_filter = LPIFilter2D(impulse_response, **filter_params)
    return predefined_filter(data)