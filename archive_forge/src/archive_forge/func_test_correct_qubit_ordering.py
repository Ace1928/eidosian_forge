import pytest
import numpy as np
import cirq
from cirq.testing.circuit_compare import _assert_apply_unitary_works_when_axes_transposed
def test_correct_qubit_ordering():
    a, b = cirq.LineQubit.range(2)
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(cirq.Circuit(cirq.Z(a), cirq.Z(b), cirq.measure(b)), cirq.Circuit(cirq.Z(a), cirq.measure(b)))
    with pytest.raises(AssertionError):
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(cirq.Circuit(cirq.Z(a), cirq.Z(b), cirq.measure(b)), cirq.Circuit(cirq.Z(b), cirq.measure(b)))