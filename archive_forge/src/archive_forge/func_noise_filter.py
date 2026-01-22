import numpy as np
from scipy import ndimage as ndi
from ..._shared.utils import check_nD, warn
from ...morphology.footprints import _footprint_is_sequence
from ...util import img_as_ubyte
from . import generic_cy
def noise_filter(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0):
    """Noise feature.

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

    References
    ----------
    .. [1] N. Hashimoto et al. Referenceless image quality evaluation
                     for whole slide imaging. J Pathol Inform 2012;3:9.

    Returns
    -------
    out : ([P,] M, N) ndarray (same dtype as input image)
        Output image.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk, ball
    >>> from skimage.filters.rank import noise_filter
    >>> import numpy as np
    >>> img = data.camera()
    >>> rng = np.random.default_rng()
    >>> volume = rng.integers(0, 255, size=(10,10,10), dtype=np.uint8)
    >>> out = noise_filter(img, disk(5))
    >>> out_vol = noise_filter(volume, ball(5))

    """
    np_image = np.asanyarray(image)
    if _footprint_is_sequence(footprint):
        raise ValueError('footprint sequences are not currently supported by rank filters')
    if np_image.ndim == 2:
        centre_r = int(footprint.shape[0] / 2) + shift_y
        centre_c = int(footprint.shape[1] / 2) + shift_x
        footprint_cpy = footprint.copy()
        footprint_cpy[centre_r, centre_c] = 0
        return _apply_scalar_per_pixel(generic_cy._noise_filter, image, footprint_cpy, out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)
    elif np_image.ndim == 3:
        centre_r = int(footprint.shape[0] / 2) + shift_y
        centre_c = int(footprint.shape[1] / 2) + shift_x
        centre_z = int(footprint.shape[2] / 2) + shift_z
        footprint_cpy = footprint.copy()
        footprint_cpy[centre_r, centre_c, centre_z] = 0
        return _apply_scalar_per_pixel_3D(generic_cy._noise_filter_3D, image, footprint_cpy, out=out, mask=mask, shift_x=shift_x, shift_y=shift_y, shift_z=shift_z)
    raise ValueError(f'`image` must have 2 or 3 dimensions, got {np_image.ndim}.')