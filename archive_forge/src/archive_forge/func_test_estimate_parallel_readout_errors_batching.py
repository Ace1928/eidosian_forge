from typing import Sequence
import pytest
import numpy as np
import cirq
def test_estimate_parallel_readout_errors_batching():
    qubits = cirq.LineQubit.range(5)
    sampler = cirq.ZerosSampler()
    repetitions = 1000
    result = cirq.estimate_parallel_single_qubit_readout_errors(sampler, qubits=qubits, repetitions=repetitions, trials=35, trials_per_batch=10)
    assert result.zero_state_errors == {q: 0.0 for q in qubits}
    assert result.one_state_errors == {q: 1.0 for q in qubits}
    assert result.repetitions == repetitions
    assert isinstance(result.timestamp, float)