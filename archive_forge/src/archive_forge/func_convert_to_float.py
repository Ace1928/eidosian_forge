import functools
import inspect
import sys
import warnings
import numpy as np
from ._warnings import all_warnings, warn
def convert_to_float(image, preserve_range):
    """Convert input image to float image with the appropriate range.

    Parameters
    ----------
    image : ndarray
        Input image.
    preserve_range : bool
        Determines if the range of the image should be kept or transformed
        using img_as_float. Also see
        https://scikit-image.org/docs/dev/user_guide/data_types.html

    Notes
    -----
    * Input images with `float32` data type are not upcast.

    Returns
    -------
    image : ndarray
        Transformed version of the input.

    """
    if image.dtype == np.float16:
        return image.astype(np.float32)
    if preserve_range:
        if image.dtype.char not in 'df':
            image = image.astype(float)
    else:
        from ..util.dtype import img_as_float
        image = img_as_float(image)
    return image