from typing import cast
import numpy as np
import pytest
import cirq
def test_act_on_stabilizer_ch_form():
    a, b = [cirq.LineQubit(3), cirq.LineQubit(1)]
    m = cirq.measure(a, b, key='out', invert_mask=(True,), confusion_map={(1,): np.array([[0, 1], [1, 0]])})
    cirq.testing.assert_all_implemented_act_on_effects_match_unitary(m)
    args = cirq.StabilizerChFormSimulationState(qubits=cirq.LineQubit.range(5), prng=np.random.RandomState(), initial_state=0)
    cirq.act_on(m, args)
    assert args.log_of_measurement_results == {'out': [1, 1]}
    args = cirq.StabilizerChFormSimulationState(qubits=cirq.LineQubit.range(5), prng=np.random.RandomState(), initial_state=8)
    cirq.act_on(m, args)
    assert args.log_of_measurement_results == {'out': [1, 0]}
    args = cirq.StabilizerChFormSimulationState(qubits=cirq.LineQubit.range(5), prng=np.random.RandomState(), initial_state=10)
    cirq.act_on(m, args)
    datastore = cast(cirq.ClassicalDataDictionaryStore, args.classical_data)
    out = cirq.MeasurementKey('out')
    assert args.log_of_measurement_results == {'out': [0, 0]}
    assert datastore.records[out] == [(0, 0)]
    cirq.act_on(m, args)
    assert args.log_of_measurement_results == {'out': [0, 0]}
    assert datastore.records[out] == [(0, 0), (0, 0)]