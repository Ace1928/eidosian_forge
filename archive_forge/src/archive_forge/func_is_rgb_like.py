import functools
import numpy as np
from .. import color
from ..util.dtype import _convert
def is_rgb_like(image, channel_axis=-1):
    """Return True if the image *looks* like it's RGB.

    This function should not be public because it is only intended to be used
    for functions that don't accept volumes as input, since checking an image's
    shape is fragile.
    """
    return image.ndim == 3 and image.shape[channel_axis] in (3, 4)