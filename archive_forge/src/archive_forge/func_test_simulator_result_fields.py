import pytest
import numpy as np
import cirq_ionq as ionq
import cirq.testing
def test_simulator_result_fields():
    result = ionq.SimulatorResult({0: 0.4, 1: 0.6}, num_qubits=1, measurement_dict={'a': [0]}, repetitions=100)
    assert result.probabilities() == {0: 0.4, 1: 0.6}
    assert result.num_qubits() == 1
    assert result.measurement_dict() == {'a': [0]}
    assert result.repetitions() == 100