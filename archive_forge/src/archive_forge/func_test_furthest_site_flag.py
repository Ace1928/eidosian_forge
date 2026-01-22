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
def test_furthest_site_flag(self):
    points = [(0, 0), (0, 1), (1, 0), (0.5, 0.5), (1.1, 1.1)]
    vor = Voronoi(points)
    assert_equal(vor.furthest_site, False)
    vor = Voronoi(points, furthest_site=True)
    assert_equal(vor.furthest_site, True)