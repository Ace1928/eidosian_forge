import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import (
from skimage.measure import CircleModel, EllipseModel, LineModelND, ransac
from skimage.measure.fit import _dynamic_max_trials
from skimage.transform import AffineTransform
def test_ransac_non_valid_best_model():
    """Example from GH issue #5572"""

    def is_model_valid(model, *random_data) -> bool:
        """Allow models with a maximum of 10 degree tilt from the vertical"""
        tilt = abs(np.arccos(np.dot(model.params[1], [0, 0, 1])))
        return tilt <= 10 / 180 * np.pi
    rng = np.random.RandomState(1)
    data = np.linspace([0, 0, 0], [0.3, 0, 1], 1000) + rng.rand(1000, 3) - 0.5
    with expected_warnings(['Estimated model is not valid']):
        ransac(data, LineModelND, min_samples=2, residual_threshold=0.3, max_trials=50, rng=0, is_model_valid=is_model_valid)