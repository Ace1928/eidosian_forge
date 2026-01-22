import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import (
from skimage.measure import CircleModel, EllipseModel, LineModelND, ransac
from skimage.measure.fit import _dynamic_max_trials
from skimage.transform import AffineTransform
def test_ransac_dynamic_max_trials():
    assert_equal(_dynamic_max_trials(100, 100, 2, 0.99), 1)
    assert_equal(_dynamic_max_trials(100, 100, 2, 1), 1)
    assert_equal(_dynamic_max_trials(95, 100, 2, 0.99), 2)
    assert_equal(_dynamic_max_trials(95, 100, 2, 1), 16)
    assert_equal(_dynamic_max_trials(90, 100, 2, 0.99), 3)
    assert_equal(_dynamic_max_trials(90, 100, 2, 1), 22)
    assert_equal(_dynamic_max_trials(70, 100, 2, 0.99), 7)
    assert_equal(_dynamic_max_trials(70, 100, 2, 1), 54)
    assert_equal(_dynamic_max_trials(50, 100, 2, 0.99), 17)
    assert_equal(_dynamic_max_trials(50, 100, 2, 1), 126)
    assert_equal(_dynamic_max_trials(95, 100, 8, 0.99), 5)
    assert_equal(_dynamic_max_trials(95, 100, 8, 1), 34)
    assert_equal(_dynamic_max_trials(90, 100, 8, 0.99), 9)
    assert_equal(_dynamic_max_trials(90, 100, 8, 1), 65)
    assert_equal(_dynamic_max_trials(70, 100, 8, 0.99), 78)
    assert_equal(_dynamic_max_trials(70, 100, 8, 1), 608)
    assert_equal(_dynamic_max_trials(50, 100, 8, 0.99), 1177)
    assert_equal(_dynamic_max_trials(50, 100, 8, 1), 9210)
    assert_equal(_dynamic_max_trials(1, 100, 5, 0), 0)
    assert_equal(_dynamic_max_trials(1, 100, 5, 1), 360436504051)