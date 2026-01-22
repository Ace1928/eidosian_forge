import cirq
import numpy as np
import pytest
def test_kraus_channel_from_channel():
    q0 = cirq.LineQubit(0)
    dp = cirq.depolarize(0.1)
    kc = cirq.KrausChannel.from_channel(dp, key='dp')
    assert cirq.measurement_key_name(kc) == 'dp'
    cirq.testing.assert_consistent_channel(kc)
    circuit = cirq.Circuit(kc.on(q0))
    sim = cirq.Simulator(seed=0)
    results = sim.simulate(circuit)
    assert 'dp' in results.measurements
    assert results.measurements['dp'] in range(4)