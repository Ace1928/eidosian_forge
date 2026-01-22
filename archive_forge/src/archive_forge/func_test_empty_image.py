import numpy as np
from skimage.morphology import convex_hull_image, convex_hull_object
from skimage.morphology._convex_hull import possible_hull
from skimage._shared import testing
from skimage._shared.testing import assert_array_equal
from skimage._shared._warnings import expected_warnings
def test_empty_image():
    image = np.zeros((6, 6), dtype=bool)
    with expected_warnings(['entirely zero']):
        assert_array_equal(convex_hull_image(image), image)