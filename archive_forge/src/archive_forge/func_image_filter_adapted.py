import functools
import numpy as np
from .. import color
from ..util.dtype import _convert
@functools.wraps(image_filter)
def image_filter_adapted(image, *args, **kwargs):
    if is_rgb_like(image):
        return apply_to_rgb(image_filter, image, *args, **kwargs)
    else:
        return image_filter(image, *args, **kwargs)