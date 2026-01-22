import pytest
from numpy.testing import assert_array_equal
import numpy as np
from skimage import graph
from skimage import segmentation, data
from skimage._shared import testing
def test_rag_boundary():
    labels = np.zeros((16, 16), dtype='uint8')
    edge_map = np.zeros_like(labels, dtype=float)
    edge_map[8, :] = 0.5
    edge_map[:, 8] = 1.0
    labels[:8, :8] = 1
    labels[:8, 8:] = 2
    labels[8:, :8] = 3
    labels[8:, 8:] = 4
    g = graph.rag_boundary(labels, edge_map, connectivity=1)
    assert set(g.nodes()) == {1, 2, 3, 4}
    assert set(g.edges()) == {(1, 2), (1, 3), (2, 4), (3, 4)}
    assert g[1][3]['weight'] == 0.25
    assert g[2][4]['weight'] == 0.34375
    assert g[1][3]['count'] == 16