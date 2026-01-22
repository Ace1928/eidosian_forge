import pytest
import numpy as np
import cirq
from cirq.testing.circuit_compare import _assert_apply_unitary_works_when_axes_transposed
def test_sensitive_to_phase():
    q = cirq.NamedQubit('q')
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(cirq.Circuit([cirq.Moment([])]), cirq.Circuit(), atol=0)
    with pytest.raises(AssertionError):
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(cirq.Circuit([cirq.Moment([cirq.Z(q) ** 0.0001])]), cirq.Circuit(), atol=0)
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(cirq.Circuit([cirq.Moment([cirq.Z(q) ** 0.0001])]), cirq.Circuit(), atol=0.01)