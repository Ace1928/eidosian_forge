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
def test_swapping(self):
    x = qhull._Qhull(b'v', np.array([[0, 0], [0, 1], [1, 0], [1, 1.0], [0.5, 0.5]]), b'Qz')
    xd = copy.deepcopy(x.get_voronoi_diagram())
    y = qhull._Qhull(b'v', np.array([[0, 0], [0, 1], [1, 0], [1, 2.0]]), b'Qz')
    yd = copy.deepcopy(y.get_voronoi_diagram())
    xd2 = copy.deepcopy(x.get_voronoi_diagram())
    x.close()
    yd2 = copy.deepcopy(y.get_voronoi_diagram())
    y.close()
    assert_raises(RuntimeError, x.get_voronoi_diagram)
    assert_raises(RuntimeError, y.get_voronoi_diagram)
    assert_allclose(xd[0], xd2[0])
    assert_unordered_tuple_list_equal(xd[1], xd2[1], tpl=sorted_tuple)
    assert_unordered_tuple_list_equal(xd[2], xd2[2], tpl=sorted_tuple)
    assert_unordered_tuple_list_equal(xd[3], xd2[3], tpl=sorted_tuple)
    assert_array_equal(xd[4], xd2[4])
    assert_allclose(yd[0], yd2[0])
    assert_unordered_tuple_list_equal(yd[1], yd2[1], tpl=sorted_tuple)
    assert_unordered_tuple_list_equal(yd[2], yd2[2], tpl=sorted_tuple)
    assert_unordered_tuple_list_equal(yd[3], yd2[3], tpl=sorted_tuple)
    assert_array_equal(yd[4], yd2[4])
    x.close()
    assert_raises(RuntimeError, x.get_voronoi_diagram)
    y.close()
    assert_raises(RuntimeError, y.get_voronoi_diagram)