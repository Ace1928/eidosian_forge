from warnings import warn
import numpy as np
from scipy import linalg
from .._shared.utils import (
from ..util import dtype, dtype_limits
@channel_as_last_axis()
def lab2xyz(lab, illuminant='D65', observer='2', *, channel_axis=-1):
    """Convert image in CIE-LAB to XYZ color space.

    Parameters
    ----------
    lab : (..., C=3, ...) array_like
        The input image in CIE-LAB color space.
        Unless `channel_axis` is set, the final dimension denotes the CIE-LAB
        channels.
        The L* values range from 0 to 100;
        the a* and b* values range from -128 to 127.
    illuminant : {"A", "B", "C", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10", "R"}, optional
        The aperture angle of the observer.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    out : (..., C=3, ...) ndarray
        The image in XYZ color space, of same shape as input.

    Raises
    ------
    ValueError
        If `lab` is not at least 2-D with shape (..., C=3, ...).
    ValueError
        If either the illuminant or the observer angle are not supported or
        unknown.
    UserWarning
        If any of the pixels are invalid (Z < 0).

    Notes
    -----
    The CIE XYZ tristimulus values are x_ref = 95.047, y_ref = 100., and
    z_ref = 108.883. See function :func:`~.xyz_tristimulus_values` for a list of
    supported illuminants.

    See Also
    --------
    xyz2lab

    References
    ----------
    .. [1] http://www.easyrgb.com/en/math.php
    .. [2] https://en.wikipedia.org/wiki/CIELAB_color_space
    """
    xyz, n_invalid = _lab2xyz(lab, illuminant, observer)
    if n_invalid != 0:
        warn(f'Conversion from CIE-LAB to XYZ color space resulted in {n_invalid} negative Z values that have been clipped to zero', stacklevel=3)
    return xyz