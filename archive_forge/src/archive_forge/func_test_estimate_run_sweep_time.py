import pytest
import cirq
import cirq_google.engine.runtime_estimator as runtime_estimator
import sympy
@pytest.mark.parametrize('depth, width, reps, sweeps, expected', [(10, 10, 1000, 1, 2.3), (10, 10, 1000, 2, 2.8), (10, 10, 1000, 4, 3.2), (10, 10, 1000, 8, 4.1), (10, 10, 1000, 16, 6.1), (10, 10, 1000, 32, 10.2), (10, 10, 1000, 64, 19.2), (40, 10, 1000, 2, 6.0), (40, 10, 1000, 4, 7.2), (40, 10, 1000, 8, 10.9), (40, 10, 1000, 16, 17.2), (40, 10, 1000, 32, 32.2), (40, 10, 1000, 64, 61.4), (40, 10, 1000, 128, 107.5), (40, 160, 32000, 32, 249.7), (30, 40, 32000, 32, 171.0), (40, 40, 32000, 32, 206.9), (40, 80, 32000, 16, 90.4), (40, 80, 32000, 8, 58.7), (40, 80, 8000, 32, 80.1), (20, 40, 32000, 32, 69.8), (30, 40, 32000, 32, 170.9), (40, 40, 32000, 32, 215.4), (2, 40, 16000, 16, 10.5), (2, 640, 16000, 16, 16.9), (2, 1280, 16000, 16, 22.6), (2, 2560, 16000, 16, 38.9)])
def test_estimate_run_sweep_time(depth, width, sweeps, reps, expected):
    """Test various run times.
    Values taken from Weber November 2021."""
    qubits = cirq.GridQubit.rect(8, 8)
    circuit = cirq.testing.random_circuit(qubits[:depth], n_moments=width, op_density=1.0)
    params = cirq.Linspace('t', 0, 1, sweeps)
    runtime = runtime_estimator.estimate_run_sweep_time(circuit, params, repetitions=reps)
    _assert_about_equal(runtime, expected)