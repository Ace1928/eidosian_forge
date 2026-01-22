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
def test_furthest_site(self):
    points = [(0, 0), (0, 1), (1, 0), (0.5, 0.5), (1.1, 1.1)]
    output = '\n        2\n        3 5 1\n        -10.101 -10.101\n        0.6000000000000001    0.5\n           0.5 0.6000000000000001\n        3 0 2 1\n        2 0 1\n        2 0 2\n        0\n        3 0 2 1\n        5\n        4 0 2 0 2\n        4 0 4 1 2\n        4 0 1 0 1\n        4 1 4 0 1\n        4 2 4 0 2\n        '
    self._compare_qvoronoi(points, output, furthest_site=True)