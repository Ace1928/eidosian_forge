import re
import textwrap
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage.transform import (
from skimage.transform._geometric import (
def test_3d_euclidean_estimation():
    src_points = np.random.rand(1000, 3)
    angles = np.random.random((3,)) * 2 * np.pi - np.pi
    rotation_matrix = _euler_rotation_matrix(angles)
    translation_vector = np.random.random((3,))
    dst_points = []
    for pt in src_points:
        pt_r = pt.reshape(3, 1)
        dst = np.matmul(rotation_matrix, pt_r) + translation_vector.reshape(3, 1)
        dst = dst.reshape(3)
        dst_points.append(dst)
    dst_points = np.array(dst_points)
    tform = EuclideanTransform(dimensionality=3)
    assert tform.estimate(src_points, dst_points)
    estimated_rotation = tform.rotation
    estimated_translation = tform.translation
    assert_almost_equal(estimated_rotation, rotation_matrix)
    assert_almost_equal(estimated_translation, translation_vector)