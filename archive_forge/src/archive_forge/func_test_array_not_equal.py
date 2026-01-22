import pytest
import numpy as np
from ase.calculators.calculator import equal
@pytest.mark.parametrize('a', arrays([2, 2], float))
@pytest.mark.parametrize('b', arrays(np.array([2, 2]) + 3.1e-08, float))
@pytest.mark.parametrize('rtol', [None, 0, 1e-08])
@pytest.mark.parametrize('atol', [None, 0, 1e-08])
def test_array_not_equal(a, b, rtol, atol):
    assert a is not b
    assert not equal(a, b, rtol=rtol, atol=atol)
    assert not equal({'size': a, 'gamma': True}, {'size': b, 'gamma': True}, rtol=rtol, atol=atol)