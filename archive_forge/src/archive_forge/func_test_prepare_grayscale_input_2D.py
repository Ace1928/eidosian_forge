import numpy as np
import pytest
from skimage._shared._dependency_checks import has_mpl
from skimage.feature.util import (
def test_prepare_grayscale_input_2D():
    with pytest.raises(ValueError):
        _prepare_grayscale_input_2D(np.zeros((3, 3, 3)))
    with pytest.raises(ValueError):
        _prepare_grayscale_input_2D(np.zeros((3, 1)))
    with pytest.raises(ValueError):
        _prepare_grayscale_input_2D(np.zeros((3, 1, 1)))
    _prepare_grayscale_input_2D(np.zeros((3, 3)))
    _prepare_grayscale_input_2D(np.zeros((3, 3, 1)))
    _prepare_grayscale_input_2D(np.zeros((1, 3, 3)))