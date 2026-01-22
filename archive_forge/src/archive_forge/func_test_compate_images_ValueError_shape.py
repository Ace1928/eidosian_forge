import numpy as np
from skimage._shared.testing import assert_array_equal
from skimage._shared import testing
from skimage.util.compare import compare_images
def test_compate_images_ValueError_shape():
    img1 = np.zeros((10, 10), dtype=np.uint8)
    img2 = np.zeros((10, 1), dtype=np.uint8)
    with testing.raises(ValueError):
        compare_images(img1, img2)