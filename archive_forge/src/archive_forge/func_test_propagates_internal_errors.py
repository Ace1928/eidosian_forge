import fractions
import pytest
import cirq
def test_propagates_internal_errors():

    class UnorderableClass:

        def __eq__(self, other):
            return NotImplemented

        def __ne__(self, other):
            return NotImplemented

        def __lt__(self, other):
            raise ValueError('oh no')

        def __le__(self, other):
            return NotImplemented

        def __ge__(self, other):
            return NotImplemented

        def __gt__(self, other):
            return NotImplemented
    ot = cirq.testing.OrderTester()
    with pytest.raises(ValueError, match='oh no'):
        ot.add_ascending(UnorderableClass())