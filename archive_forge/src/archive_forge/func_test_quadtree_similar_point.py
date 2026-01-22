import pickle
import numpy as np
import pytest
from sklearn.neighbors._quad_tree import _QuadTree
from sklearn.utils import check_random_state
def test_quadtree_similar_point():
    Xs = []
    Xs.append(np.array([[1, 2], [3, 4]], dtype=np.float32))
    Xs.append(np.array([[1.0, 2.0], [1.0, 3.0]], dtype=np.float32))
    Xs.append(np.array([[1.00001, 2.0], [1.00002, 3.0]], dtype=np.float32))
    Xs.append(np.array([[1.0, 2.0], [3.0, 2.0]], dtype=np.float32))
    Xs.append(np.array([[1.0, 2.00001], [3.0, 2.00002]], dtype=np.float32))
    Xs.append(np.array([[1.00001, 2.00001], [1.00002, 2.00002]], dtype=np.float32))
    Xs.append(np.array([[1, 0.0003817754041], [2, 0.000381775375]], dtype=np.float32))
    Xs.append(np.array([[0.0003817754041, 1.0], [0.000381775375, 2.0]], dtype=np.float32))
    for X in Xs:
        tree = _QuadTree(n_dimensions=2, verbose=0)
        tree.build_tree(X)
        tree._check_coherence()