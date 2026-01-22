import numpy as np
from scipy import ndimage as ndi
from ..._shared.utils import check_nD, warn
from ...morphology.footprints import _footprint_is_sequence
from ...util import img_as_ubyte
from . import generic_cy
def subtract_mean(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0):
    """Return image subtracted from its local mean.

    Parameters
    ----------
    image : ([P,] M, N) ndarray (uint8, uint16)
        Input image.
    footprint : ndarray
        The neighborhood expressed as an ndarray of 1's and 0's.
    out : ([P,] M, N) array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray (integer or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y, shift_z : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).

    Returns
    -------
    out : ([P,] M, N) ndarray (same dtype as input image)
        Output image.

    Notes
    -----
    Subtracting the mean value may introduce underflow. To compensate
    this potential underflow, the obtained difference is downscaled by
    a factor of 2 and shifted by `n_bins / 2 - 1`, the median value of
    the local histogram (`n_bins = max(3, image.max()) +1` for 16-bits
    images and 256 otherwise).

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk, ball
    >>> from skimage.filters.rank import subtract_mean
    >>> import numpy as np
    >>> img = data.camera()
    >>> rng = np.random.default_rng()
    >>> volume = rng.integers(0, 255, size=(10,10,10), dtype=np.uint8)
    >>> out = subtract_mean(img, disk(5))
    >>> out_vol = subtract_mean(volume, ball(5))

    """
    np_image = np.asanyarray(image)
    if np_image.ndim == 2:
        return _apply_scalar_per_pixel(generic_cy._subtract_mean, image, footprint, out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)
    elif np_image.ndim == 3:
        return _apply_scalar_per_pixel_3D(generic_cy._subtract_mean_3D, image, footprint, out=out, mask=mask, shift_x=shift_x, shift_y=shift_y, shift_z=shift_z)
    raise ValueError(f'`image` must have 2 or 3 dimensions, got {np_image.ndim}.')