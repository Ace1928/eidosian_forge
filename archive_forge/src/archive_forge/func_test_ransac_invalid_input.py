import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import (
from skimage.measure import CircleModel, EllipseModel, LineModelND, ransac
from skimage.measure.fit import _dynamic_max_trials
from skimage.transform import AffineTransform
def test_ransac_invalid_input():
    with testing.raises(ValueError):
        ransac(np.zeros((10, 2)), None, min_samples=2, residual_threshold=-0.5)
    with testing.raises(ValueError):
        ransac(np.zeros((10, 2)), None, min_samples=2, residual_threshold=0, max_trials=-1)
    with testing.raises(ValueError):
        ransac(np.zeros((10, 2)), None, min_samples=2, residual_threshold=0, stop_probability=-1)
    with testing.raises(ValueError):
        ransac(np.zeros((10, 2)), None, min_samples=2, residual_threshold=0, stop_probability=1.01)
    with testing.raises(ValueError):
        ransac(np.zeros((10, 2)), None, min_samples=0, residual_threshold=0)
    with testing.raises(ValueError):
        ransac(np.zeros((10, 2)), None, min_samples=11, residual_threshold=0)
    with testing.raises(ValueError):
        ransac(np.zeros((10, 2)), None, min_samples=-1, residual_threshold=0)