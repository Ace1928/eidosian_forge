from functools import partial
import numpy as np
from skimage import img_as_float, img_as_uint
from skimage import color, data, filters
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
def test_gray_scale_image():
    assert_allclose(edges_each(GRAY_IMAGE), filters.sobel(GRAY_IMAGE))