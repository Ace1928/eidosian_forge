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
def test_masked_array_fails(self):
    masked_array = np.ma.masked_all(1)
    assert_raises(ValueError, qhull.Voronoi, masked_array)