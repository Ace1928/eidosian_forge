import re
import textwrap
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage.transform import (
from skimage.transform._geometric import (
def test_similarity_init():
    scale = 0.1
    rotation = 1
    translation = (1, 1)
    tform = SimilarityTransform(scale=scale, rotation=rotation, translation=translation)
    assert_almost_equal(tform.scale, scale)
    assert_almost_equal(tform.rotation, rotation)
    assert_almost_equal(tform.translation, translation)
    tform2 = SimilarityTransform(tform.params)
    assert_almost_equal(tform2.scale, scale)
    assert_almost_equal(tform2.rotation, rotation)
    assert_almost_equal(tform2.translation, translation)
    scale = 0.1
    rotation = 0
    translation = (1, 1)
    tform = SimilarityTransform(scale=scale, rotation=rotation, translation=translation)
    assert_almost_equal(tform.scale, scale)
    assert_almost_equal(tform.rotation, rotation)
    assert_almost_equal(tform.translation, translation)
    scale = 0.1
    rotation = np.pi / 2
    translation = (1, 1)
    tform = SimilarityTransform(scale=scale, rotation=rotation, translation=translation)
    assert_almost_equal(tform.scale, scale)
    assert_almost_equal(tform.rotation, rotation)
    assert_almost_equal(tform.translation, translation)
    scale = 1.0
    rotation = np.pi / 2
    translation = (0, 0)
    params = np.array([[0, -1, 1.33226763e-15], [1, 2.22044605e-16, -1.33226763e-15], [0, 0, 1]])
    tform = SimilarityTransform(params)
    assert_almost_equal(tform.scale, scale)
    assert_almost_equal(tform.rotation, rotation)
    assert_almost_equal(tform.translation, translation)