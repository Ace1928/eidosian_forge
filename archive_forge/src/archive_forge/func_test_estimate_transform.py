import re
import textwrap
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage.transform import (
from skimage.transform._geometric import (
def test_estimate_transform():
    for tform in ('euclidean', 'similarity', 'affine', 'projective', 'polynomial'):
        estimate_transform(tform, SRC[:2, :], DST[:2, :])
    with pytest.raises(ValueError):
        estimate_transform('foobar', SRC[:2, :], DST[:2, :])