from warnings import warn
import numpy as np
from scipy import linalg
from .._shared.utils import (
from ..util import dtype, dtype_limits
@channel_as_last_axis()
def lch2lab(lch, *, channel_axis=-1):
    """Convert image in CIE-LCh to CIE-LAB color space.

    CIE-LCh is the cylindrical representation of the CIE-LAB (Cartesian) color
    space.

    Parameters
    ----------
    lch : (..., C=3, ...) array_like
        The input image in CIE-LCh color space.
        Unless `channel_axis` is set, the final dimension denotes the CIE-LAB
        channels.
        The L* values range from 0 to 100;
        the C values range from 0 to 100;
        the h values range from 0 to ``2*pi``.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    out : (..., C=3, ...) ndarray
        The image in CIE-LAB format, of same shape as input.

    Raises
    ------
    ValueError
        If `lch` does not have at least 3 channels (i.e., L*, C, and h).

    Notes
    -----
    The h channel (i.e., hue) is expressed as an angle in range ``(0, 2*pi)``.

    See Also
    --------
    lab2lch

    References
    ----------
    .. [1] http://www.easyrgb.com/en/math.php
    .. [2] https://en.wikipedia.org/wiki/HCL_color_space
    .. [3] https://en.wikipedia.org/wiki/CIELAB_color_space

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.color import rgb2lab, lch2lab, lab2lch
    >>> img = data.astronaut()
    >>> img_lab = rgb2lab(img)
    >>> img_lch = lab2lch(img_lab)
    >>> img_lab2 = lch2lab(img_lch)
    """
    lch = _prepare_lab_array(lch)
    c, h = (lch[..., 1], lch[..., 2])
    lch[..., 1], lch[..., 2] = (c * np.cos(h), c * np.sin(h))
    return lch