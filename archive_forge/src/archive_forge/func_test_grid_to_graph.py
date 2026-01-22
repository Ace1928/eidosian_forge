import numpy as np
import pytest
from scipy import ndimage
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.image import (
def test_grid_to_graph():
    size = 2
    roi_size = 1
    mask = np.zeros((size, size), dtype=bool)
    mask[0:roi_size, 0:roi_size] = True
    mask[-roi_size:, -roi_size:] = True
    mask = mask.reshape(size ** 2)
    A = grid_to_graph(n_x=size, n_y=size, mask=mask, return_as=np.ndarray)
    assert connected_components(A)[0] == 2
    mask = np.zeros((2, 3), dtype=bool)
    mask[0, 0] = 1
    mask[:, 2] = 1
    graph = grid_to_graph(2, 3, 1, mask=mask.ravel()).todense()
    desired = np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1]])
    np.testing.assert_array_equal(graph, desired)
    mask = np.ones((size, size), dtype=np.int16)
    A = grid_to_graph(n_x=size, n_y=size, n_z=size, mask=mask)
    assert connected_components(A)[0] == 1
    mask = np.ones((size, size))
    A = grid_to_graph(n_x=size, n_y=size, n_z=size, mask=mask, dtype=bool)
    assert A.dtype == bool
    A = grid_to_graph(n_x=size, n_y=size, n_z=size, mask=mask, dtype=int)
    assert A.dtype == int
    A = grid_to_graph(n_x=size, n_y=size, n_z=size, mask=mask, dtype=np.float64)
    assert A.dtype == np.float64