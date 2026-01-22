import re
import textwrap
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage.transform import (
from skimage.transform._geometric import (
@pytest.mark.parametrize('tform', [ProjectiveTransform(matrix=np.random.rand(3, 3)), AffineTransform(scale=(0.1, 0.1), rotation=0.3), EuclideanTransform(rotation=0.9, translation=(5, 5)), SimilarityTransform(scale=0.1, rotation=0.9), EssentialMatrixTransform(rotation=np.eye(3), translation=(1 / np.sqrt(2), 1 / np.sqrt(2), 0)), FundamentalMatrixTransform(matrix=EssentialMatrixTransform(rotation=np.eye(3), translation=(1 / np.sqrt(2), 1 / np.sqrt(2), 0)).params), (t := PiecewiseAffineTransform()).estimate(SRC, DST) and t])
def test_inverse_all_transforms(tform):
    assert isinstance(tform.inverse, type(tform))
    try:
        assert_almost_equal(tform.inverse.inverse.params, tform.params)
    except AttributeError:
        assert isinstance(tform, PiecewiseAffineTransform)
    assert_almost_equal(tform.inverse.inverse(SRC), tform(SRC))
    if not isinstance(tform, EssentialMatrixTransform | FundamentalMatrixTransform | PiecewiseAffineTransform):
        assert_almost_equal((tform + tform.inverse)(SRC), SRC)
        assert_almost_equal((tform.inverse + tform)(SRC), SRC)