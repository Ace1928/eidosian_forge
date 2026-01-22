import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_stabilizer_supports_depolarize():
    with pytest.raises(TypeError, match='act_on'):
        for _ in range(100):
            cirq.act_on(cirq.depolarize(3 / 4), ExampleSimulationState(), qubits=())
    q = cirq.LineQubit(0)
    c = cirq.Circuit(cirq.depolarize(3 / 4).on(q), cirq.measure(q, key='m'))
    m = np.sum(cirq.StabilizerSampler().sample(c, repetitions=100)['m'])
    assert 5 < m < 95