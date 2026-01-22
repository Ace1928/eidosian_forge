import numpy as np
from scipy.sparse import csc_matrix
from scipy.optimize._trustregion_constr.qp_subproblem \
from scipy.optimize._trustregion_constr.projections \
from numpy.testing import TestCase, assert_array_almost_equal, assert_equal
import pytest
def test_2d_sphere_constraints_line_intersections(self):
    ta, tb, intersect = sphere_intersections([0, 0], [1, 0], 0.5, entire_line=True)
    assert_array_almost_equal([ta, tb], [-0.5, 0.5])
    assert_equal(intersect, True)
    ta, tb, intersect = sphere_intersections([2, 0], [0, 1], 1, entire_line=True)
    assert_equal(intersect, False)
    ta, tb, intersect = sphere_intersections([2, 0], [1, 0], 1, entire_line=True)
    assert_array_almost_equal([ta, tb], [-3, -1])
    assert_equal(intersect, True)
    ta, tb, intersect = sphere_intersections([2, 0], [-1, 0], 1.5, entire_line=True)
    assert_array_almost_equal([ta, tb], [0.5, 3.5])
    assert_equal(intersect, True)
    ta, tb, intersect = sphere_intersections([2, 0], [1, 0], 2, entire_line=True)
    assert_array_almost_equal([ta, tb], [-4, 0])
    assert_equal(intersect, True)