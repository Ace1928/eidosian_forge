import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import (
from skimage.measure import CircleModel, EllipseModel, LineModelND, ransac
from skimage.measure.fit import _dynamic_max_trials
from skimage.transform import AffineTransform
def test_ransac_geometric():
    rng = np.random.default_rng(12373240)
    src = 100 * rng.random((50, 2))
    model0 = AffineTransform(scale=(0.5, 0.3), rotation=1, translation=(10, 20))
    dst = model0(src)
    outliers = (0, 5, 20)
    dst[outliers[0]] = (10000, 10000)
    dst[outliers[1]] = (-100, 100)
    dst[outliers[2]] = (50, 50)
    model_est, inliers = ransac((src, dst), AffineTransform, 2, 20, rng=rng)
    assert_almost_equal(model0.params, model_est.params)
    assert np.all(np.nonzero(inliers == False)[0] == outliers)