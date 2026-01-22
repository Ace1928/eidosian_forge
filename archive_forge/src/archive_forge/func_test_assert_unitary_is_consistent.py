import cirq
import pytest
import numpy as np
@pytest.mark.parametrize('ignore_phase', [False, True])
@pytest.mark.parametrize('g,is_consistent', [(cirq.testing.PhaseUsingCleanAncilla(theta=0.1, ancilla_bitsize=3), True), (cirq.testing.PhaseUsingDirtyAncilla(phase_state=1, ancilla_bitsize=4), True), (InconsistentGate(), False), (CleanCorrectButBorrowableIncorrectGate(use_clean_ancilla=True), True), (CleanCorrectButBorrowableIncorrectGate(use_clean_ancilla=False), False)])
def test_assert_unitary_is_consistent(g, ignore_phase, is_consistent):
    if is_consistent:
        cirq.testing.assert_unitary_is_consistent(g, ignore_phase)
        cirq.testing.assert_unitary_is_consistent(g.on(*cirq.LineQid.for_gate(g)), ignore_phase)
    else:
        with pytest.raises(AssertionError):
            cirq.testing.assert_unitary_is_consistent(g, ignore_phase)
        with pytest.raises(AssertionError):
            cirq.testing.assert_unitary_is_consistent(g.on(*cirq.LineQid.for_gate(g)), ignore_phase)