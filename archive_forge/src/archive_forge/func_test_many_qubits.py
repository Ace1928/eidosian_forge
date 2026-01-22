import pytest
import cirq
import cirq_google.engine.runtime_estimator as runtime_estimator
import sympy
@pytest.mark.parametrize('num_qubits', [54, 72, 100, 150, 200])
def test_many_qubits(num_qubits: int) -> None:
    """Regression test

    Make sure that high numbers of qubits do not
    slow the rep rate down to below zero.
    """
    qubits = cirq.LineQubit.range(num_qubits)
    sweeps_10 = cirq.Linspace('t', 0, 1, 10)
    circuit = cirq.Circuit(*[cirq.X(q) ** sympy.Symbol('t') for q in qubits], cirq.measure(*qubits))
    sweep_runtime = runtime_estimator.estimate_run_sweep_time(circuit, sweeps_10, repetitions=10000)
    assert sweep_runtime > 0