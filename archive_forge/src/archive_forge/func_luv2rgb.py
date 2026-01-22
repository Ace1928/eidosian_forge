from warnings import warn
import numpy as np
from scipy import linalg
from .._shared.utils import (
from ..util import dtype, dtype_limits
@channel_as_last_axis()
def luv2rgb(luv, *, channel_axis=-1):
    """Luv to RGB color space conversion.

    Parameters
    ----------
    luv : (..., C=3, ...) array_like
        The image in CIE Luv format. By default, the final dimension denotes
        channels.

    Returns
    -------
    out : (..., C=3, ...) ndarray
        The image in RGB format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `luv` is not at least 2-D with shape (..., C=3, ...).

    Notes
    -----
    This function uses luv2xyz and xyz2rgb.
    """
    return xyz2rgb(luv2xyz(luv))