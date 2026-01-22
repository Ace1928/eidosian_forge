from functools import partial
import numpy as np
from skimage import img_as_float, img_as_uint
from skimage import color, data, filters
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
@adapt_rgb(each_channel)
def mask_each(image, mask):
    result = image.copy()
    result[mask] = 0
    return result