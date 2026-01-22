import numpy
import pytest
import modin.numpy as np
from .utils import assert_scalar_or_array_equal
@pytest.mark.parametrize('data', [small_arr_r_2d, small_arr_r_1d], ids=['2D', '1D'])
@pytest.mark.parametrize('operator', ['isneginf', 'isposinf'])
def test_unary_without_complex(data, operator):
    x1 = data
    numpy_result = getattr(numpy, operator)(x1)
    x1 = np.array(x1)
    modin_result = getattr(np, operator)(x1)
    assert_scalar_or_array_equal(modin_result, numpy_result)