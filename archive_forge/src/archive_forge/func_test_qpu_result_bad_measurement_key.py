import pytest
import numpy as np
import cirq_ionq as ionq
import cirq.testing
def test_qpu_result_bad_measurement_key():
    result = ionq.QPUResult({0: 10, 1: 20}, num_qubits=2, measurement_dict={'a': [0], 'b': [1]})
    with pytest.raises(ValueError, match='bad'):
        result.counts('bad')