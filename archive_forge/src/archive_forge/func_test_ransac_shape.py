import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import (
from skimage.measure import CircleModel, EllipseModel, LineModelND, ransac
from skimage.measure.fit import _dynamic_max_trials
from skimage.transform import AffineTransform
def test_ransac_shape():
    model0 = CircleModel()
    model0.params = (10, 12, 3)
    t = np.linspace(0, 2 * np.pi, 1000)
    data0 = model0.predict_xy(t)
    outliers = (10, 30, 200)
    data0[outliers[0], :] = (1000, 1000)
    data0[outliers[1], :] = (-50, 50)
    data0[outliers[2], :] = (-100, -10)
    model_est, inliers = ransac(data0, CircleModel, 3, 5, rng=1)
    ransac(data0, CircleModel, 3, 5, rng=1)
    assert_almost_equal(model0.params, model_est.params)
    for outlier in outliers:
        assert outlier not in inliers