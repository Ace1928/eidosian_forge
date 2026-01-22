from typing import Optional, Sequence, Type
import pytest
import cirq
import sympy
import numpy as np
def test_composite_gates_without_matrix():

    class CompositeExample(cirq.testing.SingleQubitGate):

        def _decompose_(self, qubits):
            yield cirq.X(qubits[0])
            yield (cirq.Y(qubits[0]) ** 0.5)

    class CompositeExample2(cirq.testing.TwoQubitGate):

        def _decompose_(self, qubits):
            yield cirq.CZ(qubits[0], qubits[1])
            yield CompositeExample()(qubits[1])
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(CompositeExample()(q0), CompositeExample2()(q0, q1))
    expected = cirq.Circuit(cirq.X(q0), cirq.Y(q0) ** 0.5, cirq.CZ(q0, q1), cirq.X(q1), cirq.Y(q1) ** 0.5)
    c_new = cirq.optimize_for_target_gateset(circuit, gateset=cirq.CZTargetGateset(), ignore_failures=False)
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(c_new, expected, atol=1e-06)
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(c_new, circuit, atol=1e-06)