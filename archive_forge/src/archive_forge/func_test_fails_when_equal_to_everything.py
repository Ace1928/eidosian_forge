import fractions
import pytest
from cirq.testing.equals_tester import EqualsTester
def test_fails_when_equal_to_everything():
    eq = EqualsTester()

    class AllEqual:
        __hash__ = None

        def __eq__(self, other):
            return True

        def __ne__(self, other):
            return False
    with pytest.raises(AssertionError, match="can't be in different"):
        eq.add_equality_group(AllEqual())