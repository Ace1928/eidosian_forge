import numpy as np
from skimage.segmentation import join_segmentations, relabel_sequential
from skimage._shared import testing
from skimage._shared.testing import assert_array_equal
import pytest
@pytest.mark.parametrize('offset', (1, 5))
@pytest.mark.parametrize('with0', (False, True))
@pytest.mark.parametrize('input_starts_at_offset', (False, True))
def test_relabel_sequential_already_sequential(offset, with0, input_starts_at_offset):
    if with0:
        ar = np.array([1, 3, 0, 2, 5, 4])
    else:
        ar = np.array([1, 3, 2, 5, 4])
    if input_starts_at_offset:
        ar[ar > 0] += offset - 1
    ar_relab, fw, inv = relabel_sequential(ar, offset=offset)
    _check_maps(ar, ar_relab, fw, inv)
    if input_starts_at_offset:
        ar_relab_ref = ar
    else:
        ar_relab_ref = np.where(ar > 0, ar + offset - 1, 0)
    assert_array_equal(ar_relab, ar_relab_ref)