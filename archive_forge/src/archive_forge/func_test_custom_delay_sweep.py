import pytest
import pandas as pd
import sympy
import cirq
import cirq.experiments.t2_decay_experiment as t2
@pytest.mark.parametrize('experiment_type', [t2.ExperimentType.RAMSEY, t2.ExperimentType.HAHN_ECHO, t2.ExperimentType.CPMG])
def test_custom_delay_sweep(experiment_type):
    pulses = [1] if experiment_type == t2.ExperimentType.CPMG else None
    results = t2.t2_decay(sampler=cirq.DensityMatrixSimulator(noise=cirq.amplitude_damp(1)), qubit=cirq.GridQubit(0, 0), num_points=4, repetitions=10, min_delay=cirq.Duration(nanos=100), max_delay=cirq.Duration(micros=1), experiment_type=experiment_type, num_pulses=pulses, delay_sweep=cirq.Points('delay_ns', [1.0, 10.0, 100.0, 1000.0, 10000.0]))
    assert results == cirq.experiments.T2DecayResult(x_basis_data=pd.DataFrame(columns=['delay_ns', 0, 1], index=range(5), data=[[1.0, 10, 0], [10.0, 10, 0], [100.0, 10, 0], [1000.0, 10, 0], [10000.0, 10, 0]]), y_basis_data=pd.DataFrame(columns=['delay_ns', 0, 1], index=range(5), data=[[1.0, 10, 0], [10.0, 10, 0], [100.0, 10, 0], [1000.0, 10, 0], [10000.0, 10, 0]]))