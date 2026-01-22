import numpy as np
from scipy.sparse import csc_matrix
from scipy.optimize._trustregion_constr.qp_subproblem \
from scipy.optimize._trustregion_constr.projections \
from numpy.testing import TestCase, assert_array_almost_equal, assert_equal
import pytest
def test_2d_box_constraints(self):
    ta, tb, intersect = box_sphere_intersections([1, 1], [-2, 2], [-1, -2], [1, 2], 2, entire_line=False)
    assert_array_almost_equal([ta, tb], [0, 0.5])
    assert_equal(intersect, True)
    ta, tb, intersect = box_sphere_intersections([1, 1], [-1, 1], [-1, -3], [1, 3], 10, entire_line=False)
    assert_array_almost_equal([ta, tb], [0, 1])
    assert_equal(intersect, True)
    ta, tb, intersect = box_sphere_intersections([1, 1], [-4, 4], [-1, -3], [1, 3], 10, entire_line=False)
    assert_array_almost_equal([ta, tb], [0, 0.5])
    assert_equal(intersect, True)
    ta, tb, intersect = box_sphere_intersections([1, 1], [-4, 4], [-1, -3], [1, 3], 2, entire_line=False)
    assert_array_almost_equal([ta, tb], [0, 0.25])
    assert_equal(intersect, True)
    ta, tb, intersect = box_sphere_intersections([2, 2], [-4, 4], [-1, -3], [1, 3], 2, entire_line=False)
    assert_equal(intersect, False)
    ta, tb, intersect = box_sphere_intersections([1, 1], [-4, 4], [2, 4], [2, 4], 2, entire_line=False)
    assert_equal(intersect, False)