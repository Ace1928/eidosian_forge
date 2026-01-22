import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
def test_key_unset_in_subcircuit_outer_scope():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.measure(q0, key='a'), cirq.CircuitOperation(cirq.FrozenCircuit(cirq.X(q1).with_classical_controls('a'))))
    circuit.append(cirq.measure(q1, key='b'))
    result = cirq.Simulator().run(circuit)
    assert result.measurements['a'] == 0
    assert result.measurements['b'] == 0