import numpy
import pytest
import modin.numpy as np
from .utils import assert_scalar_or_array_equal
def test_split_2d_uneven():
    x = np.array(numpy.random.randint(-100, 100, size=(3, 2)))
    with pytest.raises(ValueError, match='array split does not result in an equal division'):
        np.split(x, 2)