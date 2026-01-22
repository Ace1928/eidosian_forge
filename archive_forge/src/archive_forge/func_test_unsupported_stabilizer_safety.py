import numpy as np
import pytest
import sympy
import cirq
def test_unsupported_stabilizer_safety():
    from cirq.protocols.act_on_protocol_test import ExampleSimulationState
    with pytest.raises(TypeError, match='act_on'):
        for _ in range(100):
            cirq.act_on(cirq.X.with_probability(0.5), ExampleSimulationState(), qubits=())
    with pytest.raises(TypeError, match='act_on'):
        cirq.act_on(cirq.X.with_probability(sympy.Symbol('x')), ExampleSimulationState(), qubits=())
    q = cirq.LineQubit(0)
    c = cirq.Circuit((cirq.X(q) ** 0.25).with_probability(0.5), cirq.measure(q, key='m'))
    with pytest.raises(TypeError, match='Failed to act'):
        cirq.StabilizerSampler().sample(c, repetitions=100)