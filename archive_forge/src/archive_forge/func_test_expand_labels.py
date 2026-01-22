from scipy import ndimage as ndi
from skimage import data
import numpy as np
from skimage import measure
from skimage.segmentation._expand_labels import expand_labels
from skimage._shared import testing
from skimage._shared.testing import assert_array_equal
@testing.parametrize('input_array, expected_output, expand_distance, spacing', [(SAMPLE1D, SAMPLE1D_EXPANDED_3, 3, 1), (SAMPLE2D, SAMPLE2D_EXPANDED_3, 3, 1), (SAMPLE2D, SAMPLE2D_EXPANDED_1_5, 1.5, 1), (EDGECASE1D, EDGECASE1D_EXPANDED_3, 3, 1), (EDGECASE2D, EDGECASE2D_EXPANDED_4, 4, 1), (SAMPLE3D, SAMPLE3D_EXPANDED_2, 2, 1), (SAMPLE3D, SAMPLE3D_EXPAND_SPACING, 1, [2, 1, 1])])
def test_expand_labels(input_array, expected_output, expand_distance, spacing):
    expanded = expand_labels(input_array, expand_distance, spacing)
    assert_array_equal(expanded, expected_output)