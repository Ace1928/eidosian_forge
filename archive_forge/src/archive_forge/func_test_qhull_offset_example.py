import numpy as np
from skimage.morphology import convex_hull_image, convex_hull_object
from skimage.morphology._convex_hull import possible_hull
from skimage._shared import testing
from skimage._shared.testing import assert_array_equal
from skimage._shared._warnings import expected_warnings
def test_qhull_offset_example():
    nonzeros = ([1367, 1368, 1368, 1368, 1369, 1369, 1369, 1369, 1369, 1370, 1370, 1370, 1370, 1370, 1370, 1370, 1371, 1371, 1371, 1371, 1371, 1371, 1371, 1371, 1371, 1372, 1372, 1372, 1372, 1372, 1372, 1372, 1372, 1372, 1373, 1373, 1373, 1373, 1373, 1373, 1373, 1373, 1373, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1375, 1375, 1375, 1375, 1375, 1376, 1376, 1376, 1377, 1372], [151, 150, 151, 152, 149, 150, 151, 152, 153, 148, 149, 150, 151, 152, 153, 154, 147, 148, 149, 150, 151, 152, 153, 154, 155, 146, 147, 148, 149, 150, 151, 152, 153, 154, 146, 147, 148, 149, 150, 151, 152, 153, 154, 147, 148, 149, 150, 151, 152, 153, 148, 149, 150, 151, 152, 149, 150, 151, 150, 155])
    image = np.zeros((1392, 1040), dtype=bool)
    image[nonzeros] = True
    expected = image.copy()
    assert_array_equal(convex_hull_image(image), expected)