import pytest
import cirq
import cirq_google.engine.runtime_estimator as runtime_estimator
import sympy
def test_estimate_run_batch_time_average_depths():
    qubits = cirq.GridQubit.rect(4, 5)
    circuit_depth_20 = cirq.testing.random_circuit(qubits, n_moments=20, op_density=1.0)
    circuit_depth_30 = cirq.testing.random_circuit(qubits, n_moments=30, op_density=1.0)
    circuit_depth_40 = cirq.testing.random_circuit(qubits, n_moments=40, op_density=1.0)
    sweeps_10 = cirq.Linspace('t', 0, 1, 10)
    sweeps_20 = cirq.Linspace('t', 0, 1, 20)
    depth_20_and_40 = runtime_estimator.estimate_run_batch_time([circuit_depth_20, circuit_depth_40], [sweeps_10, sweeps_10], repetitions=1000)
    depth_30 = runtime_estimator.estimate_run_sweep_time(circuit_depth_30, sweeps_20, repetitions=1000)
    depth_40 = runtime_estimator.estimate_run_sweep_time(circuit_depth_40, sweeps_20, repetitions=1000)
    assert depth_20_and_40 == depth_30
    assert depth_20_and_40 < depth_40