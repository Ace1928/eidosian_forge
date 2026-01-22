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
@pytest.mark.parametrize('incremental', [False, True])
def test_good3d(self, incremental):
    points = np.array([[0.0, 0.0, 0.0], [0.90029516, -0.39187448, 0.18948093], [0.4867642, -0.72627633, 0.48536925], [0.5765153, -0.81179274, -0.09285832], [0.67846893, -0.71119562, 0.1840671]])
    hull = qhull.ConvexHull(points=points, incremental=incremental, qhull_options='QG0')
    expected = np.array([True, False, False, False], dtype=bool)
    assert_equal(hull.good, expected)