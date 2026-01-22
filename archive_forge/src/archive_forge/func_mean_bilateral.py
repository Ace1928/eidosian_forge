from ..._shared.utils import check_nD
from . import bilateral_cy
from .generic import _preprocess_input
def mean_bilateral(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, s0=10, s1=10):
    """Apply a flat kernel bilateral filter.

    This is an edge-preserving and noise reducing denoising filter. It averages
    pixels based on their spatial closeness and radiometric similarity.

    Spatial closeness is measured by considering only the local pixel
    neighborhood given by a footprint (structuring element).

    Radiometric similarity is defined by the graylevel interval [g-s0, g+s1]
    where g is the current pixel graylevel.

    Only pixels belonging to the footprint and having a graylevel inside this
    interval are averaged.

    Parameters
    ----------
    image : 2-D array (uint8, uint16)
        Input image.
    footprint : 2-D array
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).
    s0, s1 : int
        Define the [s0, s1] interval around the grayvalue of the center pixel
        to be considered for computing the value.

    Returns
    -------
    out : 2-D array (same dtype as input image)
        Output image.

    See also
    --------
    skimage.restoration.denoise_bilateral

    Examples
    --------
    >>> import numpy as np
    >>> from skimage import data
    >>> from skimage.morphology import disk
    >>> from skimage.filters.rank import mean_bilateral
    >>> img = data.camera().astype(np.uint16)
    >>> bilat_img = mean_bilateral(img, disk(20), s0=10,s1=10)

    """
    return _apply(bilateral_cy._mean, image, footprint, out=out, mask=mask, shift_x=shift_x, shift_y=shift_y, s0=s0, s1=s1)