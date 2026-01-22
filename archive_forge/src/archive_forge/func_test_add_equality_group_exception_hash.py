import fractions
import pytest
from cirq.testing.equals_tester import EqualsTester
def test_add_equality_group_exception_hash():

    class FailHash:

        def __hash__(self):
            raise ValueError('injected failure')
    eq = EqualsTester()
    with pytest.raises(ValueError, match='injected failure'):
        eq.add_equality_group(FailHash())