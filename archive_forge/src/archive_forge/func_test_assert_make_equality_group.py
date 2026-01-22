import fractions
import pytest
from cirq.testing.equals_tester import EqualsTester
def test_assert_make_equality_group():
    eq = EqualsTester()
    with pytest.raises(AssertionError, match="can't be in the same"):
        eq.make_equality_group(object)
    eq.make_equality_group(lambda: 1)
    eq.make_equality_group(lambda: 2, lambda: 2.0)
    eq.add_equality_group(3)
    with pytest.raises(AssertionError, match="can't be in different"):
        eq.add_equality_group(1)
    with pytest.raises(AssertionError, match="can't be in different"):
        eq.make_equality_group(lambda: 1)
    with pytest.raises(AssertionError, match="can't be in different"):
        eq.make_equality_group(lambda: 3)