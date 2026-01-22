import pytest
import numpy as np
import pandas as pd
import sympy
import cirq
@pytest.mark.parametrize('t1', [200, 500, 700])
def test_noise_model_continous(t1):

    class GradualDecay(cirq.NoiseModel):

        def __init__(self, t1: float):
            self.t1 = t1

        def noisy_moment(self, moment, system_qubits):
            duration = max((op.gate.duration for op in moment.operations if isinstance(op.gate, cirq.WaitGate)), default=cirq.Duration(nanos=0))
            if duration > cirq.Duration(nanos=0):
                return cirq.amplitude_damp(1 - np.exp(-duration.total_nanos() / self.t1)).on_each(system_qubits)
            return moment
    results = cirq.experiments.t1_decay(sampler=cirq.DensityMatrixSimulator(noise=GradualDecay(t1)), qubit=cirq.GridQubit(0, 0), num_points=4, repetitions=10, min_delay=cirq.Duration(nanos=100), max_delay=cirq.Duration(micros=1))
    assert np.isclose(results.constant, t1, 50)