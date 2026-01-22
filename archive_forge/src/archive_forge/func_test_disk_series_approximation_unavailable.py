import numpy as np
import pytest
from numpy.testing import assert_equal
from skimage._shared.testing import fetch
from skimage.morphology import footprints
def test_disk_series_approximation_unavailable():
    with pytest.raises(ValueError):
        footprints.disk(radius=10000, decomposition='sequence')