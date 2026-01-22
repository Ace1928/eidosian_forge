import pytest
import pandas as pd
import sympy
import cirq
import cirq.experiments.t2_decay_experiment as t2
@pytest.mark.parametrize('experiment_type', [t2.ExperimentType.HAHN_ECHO, t2.ExperimentType.CPMG])
def test_spin_echo_cancels_out_constant_rate_phase(experiment_type):

    class _TimeDependentPhase(cirq.NoiseModel):

        def noisy_moment(self, moment, system_qubits):
            duration = max((op.gate.duration for op in moment.operations if isinstance(op.gate, cirq.WaitGate)), default=cirq.Duration(nanos=1))
            phase = duration.total_nanos() / 100.0
            yield (cirq.Y ** phase).on_each(system_qubits)
            yield moment
    pulses = [1] if experiment_type == t2.ExperimentType.CPMG else None
    results = cirq.experiments.t2_decay(sampler=cirq.DensityMatrixSimulator(noise=_TimeDependentPhase()), qubit=cirq.GridQubit(0, 0), num_points=10, repetitions=100, min_delay=cirq.Duration(nanos=100), max_delay=cirq.Duration(micros=1), num_pulses=pulses, experiment_type=experiment_type)
    assert (results.expectation_pauli_y['value'] < -0.8).all()