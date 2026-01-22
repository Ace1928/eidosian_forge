import pytest
import cirq
import cirq_google.engine.runtime_estimator as runtime_estimator
import sympy
def test_estimate_run_batch_time():
    qubits = cirq.GridQubit.rect(4, 5)
    circuit = cirq.testing.random_circuit(qubits[:19], n_moments=40, op_density=1.0)
    circuit2 = cirq.testing.random_circuit(qubits[:19], n_moments=40, op_density=1.0)
    circuit3 = cirq.testing.random_circuit(qubits, n_moments=40, op_density=1.0)
    sweeps_10 = cirq.Linspace('t', 0, 1, 10)
    sweeps_20 = cirq.Linspace('t', 0, 1, 20)
    sweeps_30 = cirq.Linspace('t', 0, 1, 30)
    sweeps_40 = cirq.Linspace('t', 0, 1, 40)
    sweep_runtime = runtime_estimator.estimate_run_sweep_time(circuit, sweeps_30, repetitions=1000)
    batch_runtime = runtime_estimator.estimate_run_batch_time([circuit, circuit2], [sweeps_10, sweeps_20], repetitions=1000)
    assert sweep_runtime == batch_runtime
    three_batches = runtime_estimator.estimate_run_batch_time([circuit, circuit2, circuit3], [sweeps_10, sweeps_20, sweeps_10], repetitions=1000)
    two_batches = runtime_estimator.estimate_run_batch_time([circuit, circuit3], [sweeps_30, sweeps_10], repetitions=1000)
    assert three_batches == two_batches
    sweep_runtime = runtime_estimator.estimate_run_sweep_time(circuit, sweeps_40, repetitions=1000)
    assert three_batches > sweep_runtime