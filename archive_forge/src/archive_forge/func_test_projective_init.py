import re
import textwrap
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage.transform import (
from skimage.transform._geometric import (
@pytest.mark.parametrize('array_like_input', [False, True])
def test_projective_init(array_like_input):
    tform = estimate_transform('projective', SRC, DST)
    if array_like_input:
        params = [list(p) for p in tform.params]
    else:
        params = tform.params
    tform2 = ProjectiveTransform(params)
    assert_almost_equal(tform2.params, tform.params)