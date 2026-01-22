from scipy import ndimage as ndi
from skimage import data
import numpy as np
from skimage import measure
from skimage.segmentation._expand_labels import expand_labels
from skimage._shared import testing
from skimage._shared.testing import assert_array_equal
def test_edge_case_behaviour():
    """Check edge case behavior to detect upstream changes

    For edge cases where a pixel has the same distance to several regions,
    lexicographical order seems to determine which region gets to expand
    into this pixel given the current upstream behaviour in
    scipy.ndimage.distance_map_edt.

    As a result, we expect different results when transposing the array.
    If this test fails, something has changed upstream.
    """
    expanded = expand_labels(SAMPLE_EDGECASE_BEHAVIOUR, 1)
    expanded_transpose = expand_labels(SAMPLE_EDGECASE_BEHAVIOUR.T, 1)
    assert not np.all(expanded == expanded_transpose.T)