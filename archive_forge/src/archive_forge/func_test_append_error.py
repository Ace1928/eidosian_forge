import numpy
import pytest
import modin.numpy as np
from .utils import assert_scalar_or_array_equal
@pytest.mark.xfail(reason='append error checking is incorrect: see GH#5896')
def test_append_error():
    with pytest.raises(ValueError):
        np.append(np.array([[1, 2, 3], [4, 5, 6]]), np.array([7, 8, 9]), axis=0)