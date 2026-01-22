from warnings import warn
import numpy as np
from scipy import linalg
from .._shared.utils import (
from ..util import dtype, dtype_limits
@channel_as_last_axis()
def ycbcr2rgb(ycbcr, *, channel_axis=-1):
    """YCbCr to RGB color space conversion.

    Parameters
    ----------
    ycbcr : (..., C=3, ...) array_like
        The image in YCbCr format. By default, the final dimension denotes
        channels.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    out : (..., C=3, ...) ndarray
        The image in RGB format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `ycbcr` is not at least 2-D with shape (..., C=3, ...).

    Notes
    -----
    Y is between 16 and 235. This is the color space commonly used by video
    codecs; it is sometimes incorrectly called "YUV".

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/YCbCr
    """
    arr = ycbcr.copy()
    arr[..., 0] -= 16
    arr[..., 1] -= 128
    arr[..., 2] -= 128
    return _convert(rgb_from_ycbcr, arr)