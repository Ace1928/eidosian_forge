import numpy as np
from skimage._shared import testing
from skimage._shared.testing import assert_equal
from skimage.util._label import label_points
def test_label_points_coords_dimension():
    coords, output_shape = (np.array([[1, 2], [3, 4]]), (5, 5, 2))
    with testing.raises(ValueError):
        label_points(coords, output_shape)