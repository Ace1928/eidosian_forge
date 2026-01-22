import numpy as np
import pytest
import cirq
import cirq_google
from cirq_google.api import v2
def test_multiple_measurements_different_slots():
    circuit = cirq.Circuit()
    circuit.append(cirq.measure(q(0, 0), q(0, 1), key='k0'))
    circuit.append(cirq.measure(q(0, 2), q(0, 0), key='k1'))
    measurements = v2.find_measurements(circuit)
    assert len(measurements) == 2
    m0, m1 = measurements
    _check_measurement(m0, 'k0', [q(0, 0), q(0, 1)], 1)
    _check_measurement(m1, 'k1', [q(0, 2), q(0, 0)], 1)