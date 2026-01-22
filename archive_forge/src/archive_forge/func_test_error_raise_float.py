import numpy as np
import pytest
from skimage._shared.testing import expected_warnings, run_in_parallel
from skimage.feature import (
from skimage.transform import integral_image
def test_error_raise_float(self):
    for dtype in [float, np.double, np.float16, np.float32, np.float64]:
        with pytest.raises(ValueError):
            graycomatrix(self.image.astype(dtype), [1], [np.pi], 4)