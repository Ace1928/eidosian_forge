from warnings import warn
import numpy as np
from scipy import linalg
from .._shared.utils import (
from ..util import dtype, dtype_limits
@channel_as_last_axis()
def lab2lch(lab, *, channel_axis=-1):
    """Convert image in CIE-LAB to CIE-LCh color space.

    CIE-LCh is the cylindrical representation of the CIE-LAB (Cartesian) color
    space.

    Parameters
    ----------
    lab : (..., C=3, ...) array_like
        The input image in CIE-LAB color space.
        Unless `channel_axis` is set, the final dimension denotes the CIE-LAB
        channels.
        The L* values range from 0 to 100;
        the a* and b* values range from -128 to 127.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    out : (..., C=3, ...) ndarray
        The image in CIE-LCh color space, of same shape as input.

    Raises
    ------
    ValueError
        If `lab` does not have at least 3 channels (i.e., L*, a*, and b*).

    Notes
    -----
    The h channel (i.e., hue) is expressed as an angle in range ``(0, 2*pi)``.

    See Also
    --------
    lch2lab

    References
    ----------
    .. [1] http://www.easyrgb.com/en/math.php
    .. [2] https://en.wikipedia.org/wiki/CIELAB_color_space
    .. [3] https://en.wikipedia.org/wiki/HCL_color_space

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.color import rgb2lab, lab2lch
    >>> img = data.astronaut()
    >>> img_lab = rgb2lab(img)
    >>> img_lch = lab2lch(img_lab)
    """
    lch = _prepare_lab_array(lab)
    a, b = (lch[..., 1], lch[..., 2])
    lch[..., 1], lch[..., 2] = _cart2polar_2pi(a, b)
    return lch