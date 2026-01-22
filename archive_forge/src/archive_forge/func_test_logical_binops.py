import numpy
import pytest
import modin.numpy as np
from .utils import assert_scalar_or_array_equal
@pytest.mark.parametrize('operand1_shape', [100, (3, 100)])
@pytest.mark.parametrize('operand2_shape', [100, (3, 100)])
@pytest.mark.parametrize('operator', ['logical_and', 'logical_or', 'logical_xor'])
def test_logical_binops(operand1_shape, operand2_shape, operator):
    if operand1_shape != operand2_shape:
        pytest.xfail('TODO fix broadcasting behavior for binary logic operators')
    x1 = numpy.random.randint(-100, 100, size=operand1_shape)
    x2 = numpy.random.randint(-100, 100, size=operand2_shape)
    numpy_result = getattr(numpy, operator)(x1, x2)
    x1, x2 = (np.array(x1), np.array(x2))
    modin_result = getattr(np, operator)(x1, x2)
    assert_scalar_or_array_equal(modin_result, numpy_result, err_msg=f'Logic binary operator {operator} failed.')