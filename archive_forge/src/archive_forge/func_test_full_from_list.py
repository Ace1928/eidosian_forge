import sys
import numpy as np
from numpy.core._rational_tests import rational
import pytest
from numpy.testing import (
@pytest.mark.parametrize(['shape', 'fill_value', 'expected_output'], [((2, 2), [5.0, 6.0], np.array([[5.0, 6.0], [5.0, 6.0]])), ((3, 2), [1.0, 2.0], np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]))])
def test_full_from_list(shape, fill_value, expected_output):
    output = np.full(shape, fill_value)
    assert_equal(output, expected_output)