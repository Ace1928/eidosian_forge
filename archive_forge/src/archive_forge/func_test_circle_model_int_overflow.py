import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import (
from skimage.measure import CircleModel, EllipseModel, LineModelND, ransac
from skimage.measure.fit import _dynamic_max_trials
from skimage.transform import AffineTransform
def test_circle_model_int_overflow():
    xy = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.int32)
    xy += 500
    model = CircleModel()
    model.estimate(xy)
    assert_almost_equal(model.params, [500, 500, 1])