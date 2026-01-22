import numpy as np
import pytest
from scipy import ndimage
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.image import (
def test_img_to_graph():
    x, y = np.mgrid[:4, :4] - 10
    grad_x = img_to_graph(x)
    grad_y = img_to_graph(y)
    assert grad_x.nnz == grad_y.nnz
    np.testing.assert_array_equal(grad_x.data[grad_x.data > 0], grad_y.data[grad_y.data > 0])