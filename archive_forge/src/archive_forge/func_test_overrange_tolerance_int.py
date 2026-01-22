import numpy as np
import pytest
from skimage.morphology import flood, flood_fill
@pytest.mark.parametrize('tolerance', [-150, 150, -379, 379])
def test_overrange_tolerance_int(tolerance):
    image = np.arange(256, dtype=np.uint8).reshape((8, 8, 4))
    seed = (3, 4, 2)
    expected = np.zeros_like(image)
    output = flood_fill(image, seed, 0, tolerance=tolerance)
    np.testing.assert_equal(output, expected)