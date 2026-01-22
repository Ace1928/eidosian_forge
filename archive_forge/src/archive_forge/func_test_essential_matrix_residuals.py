import re
import textwrap
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage.transform import (
from skimage.transform._geometric import (
def test_essential_matrix_residuals():
    tform = EssentialMatrixTransform(rotation=np.eye(3), translation=np.array([1, 0, 0]))
    src = np.array([[0, 0], [0, 0], [0, 0]])
    dst = np.array([[2, 0], [2, 1], [2, 2]])
    assert_almost_equal(tform.residuals(src, dst) ** 2, [0, 0.5, 2])