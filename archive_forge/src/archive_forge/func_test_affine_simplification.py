import copy
import numpy as np
from numpy.testing import (assert_allclose, assert_almost_equal,
import pytest
from matplotlib import scale
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from matplotlib.transforms import Affine2D, Bbox, TransformedBbox
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_affine_simplification(self):
    points = np.array([[0, 0], [10, 20], [np.nan, 1], [-1, 0]], dtype=np.float64)
    na_pts = self.stack1.transform_non_affine(points)
    all_pts = self.stack1.transform(points)
    na_expected = np.array([[1.0, 2.0], [-19.0, 12.0], [np.nan, np.nan], [1.0, 1.0]], dtype=np.float64)
    all_expected = np.array([[11.0, 4.0], [-9.0, 24.0], [np.nan, np.nan], [11.0, 2.0]], dtype=np.float64)
    assert_array_almost_equal(na_pts, na_expected)
    assert_array_almost_equal(all_pts, all_expected)
    assert_array_almost_equal(self.stack1.transform_affine(na_pts), all_expected)
    assert_array_almost_equal(self.stack1.get_affine().transform(na_pts), all_expected)
    expected_result = (self.ta2 + self.ta3).get_matrix()
    result = self.stack1.get_affine().get_matrix()
    assert_array_equal(expected_result, result)
    result = self.stack2.get_affine().get_matrix()
    assert_array_equal(expected_result, result)