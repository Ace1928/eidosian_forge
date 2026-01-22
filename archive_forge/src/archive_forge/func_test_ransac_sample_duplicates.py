import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import (
from skimage.measure import CircleModel, EllipseModel, LineModelND, ransac
from skimage.measure.fit import _dynamic_max_trials
from skimage.transform import AffineTransform
def test_ransac_sample_duplicates():

    class DummyModel:
        """Dummy model to check for duplicates."""

        def estimate(self, data):
            assert_equal(np.unique(data).size, data.size)
            return True

        def residuals(self, data):
            return np.ones(len(data), dtype=np.float64)
    data = np.arange(4)
    with expected_warnings(['No inliers found']):
        ransac(data, DummyModel, min_samples=3, residual_threshold=0.0, max_trials=10)