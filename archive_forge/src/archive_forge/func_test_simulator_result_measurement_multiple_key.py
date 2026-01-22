import pytest
import numpy as np
import cirq_ionq as ionq
import cirq.testing
def test_simulator_result_measurement_multiple_key():
    result = ionq.SimulatorResult({0: 0.2, 1: 0.8}, num_qubits=2, measurement_dict={'a': [0], 'b': [1]}, repetitions=100)
    assert result.probabilities('a') == {0: 1.0}
    assert result.probabilities('b') == {0: 0.2, 1: 0.8}
    result = ionq.SimulatorResult({0: 0.2, 1: 0.8}, num_qubits=2, measurement_dict={'a': [1], 'b': [0]}, repetitions=100)
    assert result.probabilities('a') == {0: 0.2, 1: 0.8}
    assert result.probabilities('b') == {0: 1.0}