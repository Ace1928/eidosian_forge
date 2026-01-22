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
def test_volume_area(self):
    points = np.array([(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0), (0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)])
    tri = qhull.ConvexHull(points)
    assert_allclose(tri.volume, 1.0, rtol=1e-14)
    assert_allclose(tri.area, 6.0, rtol=1e-14)