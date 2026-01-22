import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import (
from skimage.measure import CircleModel, EllipseModel, LineModelND, ransac
from skimage.measure.fit import _dynamic_max_trials
from skimage.transform import AffineTransform
def test_ransac_with_no_final_inliers():
    data = np.random.rand(5, 2)
    with expected_warnings(['No inliers found. Model not fitted']):
        model, inliers = ransac(data, model_class=LineModelND, min_samples=3, residual_threshold=0, rng=1523427)
    assert inliers is None
    assert model is None