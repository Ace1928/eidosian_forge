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
def test_joggle(self):
    points = np.random.rand(10, 2)
    points = np.r_[points, points]
    tri = qhull.Delaunay(points, qhull_options='QJ Qbb Pp')
    assert_array_equal(np.unique(tri.simplices.ravel()), np.arange(len(points)))