import pytest
import numpy as np
import cirq
from cirq.testing.circuit_compare import _assert_apply_unitary_works_when_axes_transposed
def test_measuring_qubits():
    a, b = cirq.LineQubit.range(2)
    with pytest.raises(AssertionError):
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(cirq.Circuit([cirq.Moment([cirq.measure(a)])]), cirq.Circuit([cirq.Moment([cirq.measure(b)])]))
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(cirq.Circuit([cirq.Moment([cirq.measure(a, b, invert_mask=(True,))])]), cirq.Circuit([cirq.Moment([cirq.measure(b, a, invert_mask=(False, True))])]))
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(cirq.Circuit([cirq.Moment([cirq.measure(a)]), cirq.Moment([cirq.measure(b)])]), cirq.Circuit([cirq.Moment([cirq.measure(a, b)])]))