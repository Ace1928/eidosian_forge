import pytest
import numpy as np
import cirq_ionq as ionq
import cirq.testing
def test_qpu_result_fields():
    result = ionq.QPUResult({0: 10, 1: 10}, num_qubits=1, measurement_dict={'a': [0]})
    assert result.counts() == {0: 10, 1: 10}
    assert result.repetitions() == 20
    assert result.num_qubits() == 1
    assert result.measurement_dict() == {'a': [0]}