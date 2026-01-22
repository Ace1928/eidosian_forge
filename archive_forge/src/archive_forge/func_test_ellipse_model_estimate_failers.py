import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import (
from skimage.measure import CircleModel, EllipseModel, LineModelND, ransac
from skimage.measure.fit import _dynamic_max_trials
from skimage.transform import AffineTransform
@xfail(condition=arch32, reason='Known test failure on 32-bit platforms. See links for details: https://github.com/scikit-image/scikit-image/issues/3091 https://github.com/scikit-image/scikit-image/issues/2670')
def test_ellipse_model_estimate_failers():
    model = EllipseModel()
    warning_message = 'Standard deviation of data is too small to estimate ellipse with meaningful precision.'
    with pytest.warns(RuntimeWarning, match=warning_message) as _warnings:
        assert not model.estimate(np.ones((6, 2)))
    assert_stacklevel(_warnings)
    assert len(_warnings) == 1
    assert not model.estimate(np.array([[50, 80], [51, 81], [52, 80]]))