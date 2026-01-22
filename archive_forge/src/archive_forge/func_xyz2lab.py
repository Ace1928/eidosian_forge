from warnings import warn
import numpy as np
from scipy import linalg
from .._shared.utils import (
from ..util import dtype, dtype_limits
@channel_as_last_axis()
def xyz2lab(xyz, illuminant='D65', observer='2', *, channel_axis=-1):
    """XYZ to CIE-LAB color space conversion.

    Parameters
    ----------
    xyz : (..., C=3, ...) array_like
        The image in XYZ format. By default, the final dimension denotes
        channels.
    illuminant : {"A", "B", "C", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10", "R"}, optional
        One of: 2-degree observer, 10-degree observer, or 'R' observer as in
        R function grDevices::convertColor.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    out : (..., C=3, ...) ndarray
        The image in CIE-LAB format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `xyz` is not at least 2-D with shape (..., C=3, ...).
    ValueError
        If either the illuminant or the observer angle is unsupported or
        unknown.

    Notes
    -----
    By default Observer="2", Illuminant="D65". CIE XYZ tristimulus values
    x_ref=95.047, y_ref=100., z_ref=108.883. See function
    :func:`~.xyz_tristimulus_values` for a list of supported illuminants.

    References
    ----------
    .. [1] http://www.easyrgb.com/en/math.php
    .. [2] https://en.wikipedia.org/wiki/CIELAB_color_space

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.color import rgb2xyz, xyz2lab
    >>> img = data.astronaut()
    >>> img_xyz = rgb2xyz(img)
    >>> img_lab = xyz2lab(img_xyz)
    """
    arr = _prepare_colorarray(xyz, channel_axis=-1)
    xyz_ref_white = xyz_tristimulus_values(illuminant=illuminant, observer=observer, dtype=arr.dtype)
    arr = arr / xyz_ref_white
    mask = arr > 0.008856
    arr[mask] = np.cbrt(arr[mask])
    arr[~mask] = 7.787 * arr[~mask] + 16.0 / 116.0
    x, y, z = (arr[..., 0], arr[..., 1], arr[..., 2])
    L = 116.0 * y - 16.0
    a = 500.0 * (x - y)
    b = 200.0 * (y - z)
    return np.concatenate([x[..., np.newaxis] for x in [L, a, b]], axis=-1)