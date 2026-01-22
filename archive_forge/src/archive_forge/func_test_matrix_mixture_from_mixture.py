import cirq
import numpy as np
import pytest
def test_matrix_mixture_from_mixture():
    q0 = cirq.LineQubit(0)
    dp = cirq.depolarize(0.1)
    mm = cirq.MixedUnitaryChannel.from_mixture(dp, key='dp')
    assert cirq.measurement_key_name(mm) == 'dp'
    cirq.testing.assert_consistent_channel(mm)
    cirq.testing.assert_consistent_mixture(mm)
    circuit = cirq.Circuit(mm.on(q0))
    sim = cirq.Simulator(seed=0)
    results = sim.simulate(circuit)
    assert 'dp' in results.measurements
    assert results.measurements['dp'] in range(4)