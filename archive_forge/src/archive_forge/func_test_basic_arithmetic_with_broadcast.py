import numpy
import pytest
import modin.numpy as np
from .utils import assert_scalar_or_array_equal
@pytest.mark.parametrize('operand1_shape', [100, (1, 100), (3, 100)])
@pytest.mark.parametrize('operand2_shape', [100, (1, 100), (3, 100), 1])
@pytest.mark.parametrize('operator', ['__add__', '__sub__', '__truediv__', '__mul__', '__rtruediv__', '__rmul__', '__radd__', '__rsub__', '__ge__', '__gt__', '__lt__', '__le__', '__eq__', '__ne__'])
def test_basic_arithmetic_with_broadcast(operand1_shape, operand2_shape, operator):
    """Test of operators that support broadcasting."""
    if operand1_shape == (1, 100) or operand2_shape == (1, 100):
        pytest.xfail(reason='broadcasting is broken: see GH#5894')
    operand1 = numpy.random.randint(-100, 100, size=operand1_shape)
    operand2 = numpy.random.randint(-100, 100, size=operand2_shape)
    numpy_result = getattr(operand1, operator)(operand2)
    if operand2_shape == 1:
        modin_result = getattr(np.array(operand1), operator)(operand2[0])
    else:
        modin_result = getattr(np.array(operand1), operator)(np.array(operand2))
    if operator not in ['__truediv__', '__rtruediv__']:
        assert_scalar_or_array_equal(modin_result, numpy_result, err_msg=f'Binary Op {operator} failed.')
    else:
        numpy.testing.assert_array_almost_equal(modin_result._to_numpy(), numpy_result, decimal=12, err_msg='Binary Op __truediv__ failed.')