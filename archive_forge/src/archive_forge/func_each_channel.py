import functools
import numpy as np
from .. import color
from ..util.dtype import _convert
def each_channel(image_filter, image, *args, **kwargs):
    """Return color image by applying `image_filter` on channels of `image`.

    Note that this function is intended for use with `adapt_rgb`.

    Parameters
    ----------
    image_filter : function
        Function that filters a gray-scale image.
    image : array
        Input image.
    """
    c_new = [image_filter(c, *args, **kwargs) for c in np.moveaxis(image, -1, 0)]
    return np.stack(c_new, axis=-1)