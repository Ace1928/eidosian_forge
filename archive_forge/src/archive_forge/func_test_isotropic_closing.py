import numpy as np
from numpy.testing import assert_array_equal
from skimage import color, data, morphology
from skimage.morphology import binary, isotropic
from skimage.util import img_as_bool
def test_isotropic_closing():
    isotropic_res = isotropic.isotropic_closing(bw_img, 3)
    binary_res = img_as_bool(binary.binary_closing(bw_img, morphology.disk(3)))
    assert_array_equal(isotropic_res, binary_res)