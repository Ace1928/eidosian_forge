import fractions
import pytest
from cirq.testing.equals_tester import EqualsTester
def test_fails_when_not_commutative():
    eq = EqualsTester()

    class NotCommutativeImplementation:

        def __init__(self, x):
            self.x = x

        def __eq__(self, other):
            if not isinstance(other, type(self)):
                return NotImplemented
            return self.x <= other.x

        def __ne__(self, other):
            return not self == other
    with pytest.raises(AssertionError, match="can't be in the same"):
        eq.add_equality_group(NotCommutativeImplementation(0), NotCommutativeImplementation(1))
    with pytest.raises(AssertionError, match="can't be in the same"):
        eq.add_equality_group(NotCommutativeImplementation(1), NotCommutativeImplementation(0))