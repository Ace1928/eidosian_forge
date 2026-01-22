import numpy as np
import pytest
from numpy.testing import assert_equal
from skimage._shared.testing import fetch
from skimage.morphology import footprints
@pytest.mark.parametrize('radius', [1, 2, 3, 4, 5, 10, 20, 50, 75])
@pytest.mark.parametrize('strict_radius', [False, True])
def test_disk_crosses_approximation(radius, strict_radius):
    fp_func = footprints.disk
    expected = fp_func(radius, strict_radius=strict_radius, decomposition=None)
    footprint_sequence = fp_func(radius, strict_radius=strict_radius, decomposition='crosses')
    approximate = footprints.footprint_from_sequence(footprint_sequence)
    assert approximate.shape == expected.shape
    error = np.sum(np.abs(expected.astype(int) - approximate.astype(int)))
    max_error = 0.05
    assert error / expected.size <= max_error