from ..._shared.utils import check_nD
from . import percentile_cy
from .generic import _preprocess_input
def threshold_percentile(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, p0=0):
    """Local threshold of an image.

    The resulting binary mask is True if the grayvalue of the center pixel is
    greater than the local mean.

    Only grayvalues between percentiles [p0, p1] are considered in the filter.

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
    p0 : float in [0, ..., 1]
        Set the percentile value.

    Returns
    -------
    out : 2-D array (same dtype as input image)
        Output image.

    """
    return _apply(percentile_cy._threshold, image, footprint, out=out, mask=mask, shift_x=shift_x, shift_y=shift_y, p0=p0, p1=0)