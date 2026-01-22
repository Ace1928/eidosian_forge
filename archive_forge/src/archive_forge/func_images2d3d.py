import numpy as np
from skimage.morphology import convex_hull_image, convex_hull_object
from skimage.morphology._convex_hull import possible_hull
from skimage._shared import testing
from skimage._shared.testing import assert_array_equal
from skimage._shared._warnings import expected_warnings
@testing.fixture
def images2d3d():
    from ...measure.tests.test_regionprops import SAMPLE as image
    image3d = np.stack((image, image, image))
    return (image, image3d)