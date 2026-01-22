import pytest
import numpy as np
import cirq
def test_assert_phase_by_is_consistent_with_unitary():
    cirq.testing.assert_phase_by_is_consistent_with_unitary(GoodPhaser(0.5))
    cirq.testing.assert_phase_by_is_consistent_with_unitary(GoodQuditPhaser(0.5))
    with pytest.raises(AssertionError, match='Phased unitary was incorrect for index #0'):
        cirq.testing.assert_phase_by_is_consistent_with_unitary(BadPhaser(0.5))
    with pytest.raises(AssertionError, match='Phased unitary was incorrect for index #1'):
        cirq.testing.assert_phase_by_is_consistent_with_unitary(SemiBadPhaser([0.5, 0.25]))
    cirq.testing.assert_phase_by_is_consistent_with_unitary(NotPhaser())