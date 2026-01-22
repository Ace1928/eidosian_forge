import pytest
import pandas as pd
import sympy
import cirq
import cirq.experiments.t2_decay_experiment as t2
def test_sudden_decay_results():

    class _SuddenDecay(cirq.NoiseModel):

        def noisy_moment(self, moment, system_qubits):
            duration = max((op.gate.duration for op in moment.operations if isinstance(op.gate, cirq.WaitGate)), default=cirq.Duration())
            if duration > cirq.Duration(nanos=500):
                yield cirq.amplitude_damp(1).on_each(system_qubits)
            yield moment
    results = cirq.experiments.t2_decay(sampler=cirq.DensityMatrixSimulator(noise=_SuddenDecay()), qubit=cirq.GridQubit(0, 0), num_points=4, repetitions=500, min_delay=cirq.Duration(nanos=100), max_delay=cirq.Duration(micros=1))
    assert (results.expectation_pauli_y['value'][0:2] == -1).all()
    assert (results.expectation_pauli_y['value'][2:] < 0.2).all()
    assert (abs(results.expectation_pauli_x['value']) < 0.2).all()