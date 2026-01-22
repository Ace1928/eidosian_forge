import numpy as np
import pytest
import cirq
def test_fail_fast_measure_large_memory():
    num_qubits = 100
    measurement_op = cirq.MeasurementGate(num_qubits, 'a').on(*cirq.LineQubit.range(num_qubits))
    assert not cirq.has_unitary(measurement_op)