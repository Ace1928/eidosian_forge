from typing import Optional, List
import pytest
import cirq
from cirq.transformers.transformer_primitives import MAPPED_CIRCUIT_OP_TAG
def test_merge_moments_empty_moment_as_intermediate_step():
    q = cirq.NamedQubit('q')
    c_orig = cirq.Circuit([cirq.X(q), cirq.Y(q), cirq.Z(q)] * 2, cirq.X(q) ** 0.5)

    def merge_func(m1: cirq.Moment, m2: cirq.Moment):
        gate = cirq.single_qubit_matrix_to_phxz(cirq.unitary(cirq.Circuit(m1, m2)), atol=1e-08)
        return cirq.Moment(gate.on(q) if gate else [])
    c_new = cirq.merge_moments(c_orig, merge_func)
    assert len(c_new) == 1
    assert isinstance(c_new[0][q].gate, cirq.PhasedXZGate)
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(c_orig, c_new, atol=1e-08)