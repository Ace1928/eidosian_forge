import fractions
import pytest
import cirq
def test_fails_on_inconsistent_hashes():

    class ModifiedHash(tuple):

        def __hash__(self):
            return super().__hash__() ^ 1
    ot = cirq.testing.OrderTester()
    ot.add_ascending((1, 0), (1, 1))
    ot.add_ascending(ModifiedHash((1, 2)), ModifiedHash((2, 0)))
    ot.add_ascending_equivalence_group((2, 2), (2, 2))
    ot.add_ascending_equivalence_group(ModifiedHash((3, 3)), ModifiedHash((3, 3)))
    with pytest.raises(AssertionError, match='different hashes'):
        ot.add_ascending_equivalence_group((4, 4), ModifiedHash((4, 4)))