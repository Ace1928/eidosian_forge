import numpy as np
import pytest
from numpy.testing import assert_equal
from skimage._shared.testing import fetch
from skimage.morphology import footprints
def test_ball_series_approximation_unavailable():
    with pytest.raises(ValueError):
        footprints.ball(radius=10000, decomposition='sequence')