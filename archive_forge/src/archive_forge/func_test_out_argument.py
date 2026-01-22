import numpy as np
from numpy.testing import assert_array_equal
from skimage import color, data, morphology
from skimage.morphology import binary, isotropic
from skimage.util import img_as_bool
def test_out_argument():
    for func in (isotropic.isotropic_erosion, isotropic.isotropic_dilation):
        radius = 3
        img = np.ones((10, 10))
        out = np.zeros_like(img)
        out_saved = out.copy()
        func(img, radius, out=out)
        assert np.any(out != out_saved)
        assert_array_equal(out, func(img, radius))