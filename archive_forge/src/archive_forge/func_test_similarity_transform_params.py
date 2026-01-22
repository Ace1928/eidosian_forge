import re
import textwrap
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage.transform import (
from skimage.transform._geometric import (
def test_similarity_transform_params():
    with pytest.raises(ValueError):
        _ = SimilarityTransform(translation=(4, 5, 6, 7), dimensionality=4)
    tf = SimilarityTransform(scale=4, dimensionality=3)
    assert_equal(tf([[1, 1, 1]]), [[4, 4, 4]])