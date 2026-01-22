import os
import copy
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
import scipy.spatial._qhull as qhull
from scipy.spatial import cKDTree as KDTree
from scipy.spatial import Voronoi
import itertools
def test_find_simplex(self):
    points = np.array([(0, 0), (0, 1), (1, 1), (1, 0)], dtype=np.float64)
    tri = qhull.Delaunay(points)
    assert_equal(tri.simplices, [[1, 3, 2], [3, 1, 0]])
    for p in [(0.25, 0.25, 1), (0.75, 0.75, 0), (0.3, 0.2, 1)]:
        i = tri.find_simplex(p[:2])
        assert_equal(i, p[2], err_msg=f'{p!r}')
        j = qhull.tsearch(tri, p[:2])
        assert_equal(i, j)