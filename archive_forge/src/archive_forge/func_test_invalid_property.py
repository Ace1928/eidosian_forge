import numpy as np
import pytest
from skimage._shared.testing import expected_warnings, run_in_parallel
from skimage.feature import (
from skimage.transform import integral_image
def test_invalid_property(self):
    result = graycomatrix(self.image, [1], [0], 4)
    with pytest.raises(ValueError):
        graycoprops(result, 'ABC')