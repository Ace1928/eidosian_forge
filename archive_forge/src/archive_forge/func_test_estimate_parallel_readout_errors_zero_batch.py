from typing import Sequence
import pytest
import numpy as np
import cirq
def test_estimate_parallel_readout_errors_zero_batch():
    qubits = cirq.LineQubit.range(10)
    with pytest.raises(ValueError, match='non-zero trials_per_batch'):
        _ = cirq.estimate_parallel_single_qubit_readout_errors(cirq.ZerosSampler(), qubits=qubits, repetitions=1000, trials=10, trials_per_batch=0)