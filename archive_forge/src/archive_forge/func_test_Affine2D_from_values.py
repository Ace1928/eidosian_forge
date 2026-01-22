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
def test_Affine2D_from_values():
    points = np.array([[0, 0], [10, 20], [-1, 0]])
    t = mtransforms.Affine2D.from_values(1, 0, 0, 0, 0, 0)
    actual = t.transform(points)
    expected = np.array([[0, 0], [10, 0], [-1, 0]])
    assert_almost_equal(actual, expected)
    t = mtransforms.Affine2D.from_values(0, 2, 0, 0, 0, 0)
    actual = t.transform(points)
    expected = np.array([[0, 0], [0, 20], [0, -2]])
    assert_almost_equal(actual, expected)
    t = mtransforms.Affine2D.from_values(0, 0, 3, 0, 0, 0)
    actual = t.transform(points)
    expected = np.array([[0, 0], [60, 0], [0, 0]])
    assert_almost_equal(actual, expected)
    t = mtransforms.Affine2D.from_values(0, 0, 0, 4, 0, 0)
    actual = t.transform(points)
    expected = np.array([[0, 0], [0, 80], [0, 0]])
    assert_almost_equal(actual, expected)
    t = mtransforms.Affine2D.from_values(0, 0, 0, 0, 5, 0)
    actual = t.transform(points)
    expected = np.array([[5, 0], [5, 0], [5, 0]])
    assert_almost_equal(actual, expected)
    t = mtransforms.Affine2D.from_values(0, 0, 0, 0, 0, 6)
    actual = t.transform(points)
    expected = np.array([[0, 6], [0, 6], [0, 6]])
    assert_almost_equal(actual, expected)