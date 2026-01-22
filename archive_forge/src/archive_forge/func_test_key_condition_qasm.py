import re
import pytest
import sympy
import cirq
def test_key_condition_qasm():
    with pytest.raises(ValueError, match='QASM is defined only for SympyConditions'):
        _ = cirq.KeyCondition(cirq.MeasurementKey('a')).qasm