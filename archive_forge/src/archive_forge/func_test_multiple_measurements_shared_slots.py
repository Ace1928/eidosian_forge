import numpy as np
import pytest
import cirq
import cirq_google
from cirq_google.api import v2
def test_multiple_measurements_shared_slots():
    circuit = cirq.Circuit()
    circuit.append([cirq.measure(q(0, 0), q(0, 1), key='k0'), cirq.measure(q(0, 2), q(1, 1), key='k1')])
    circuit.append([cirq.measure(q(1, 0), q(0, 0), q(0, 1), key='k2'), cirq.measure(q(1, 1), q(0, 2), key='k3')])
    measurements = v2.find_measurements(circuit)
    assert len(measurements) == 4
    m0, m1, m2, m3 = measurements
    _check_measurement(m0, 'k0', [q(0, 0), q(0, 1)], 1)
    _check_measurement(m1, 'k1', [q(0, 2), q(1, 1)], 1)
    _check_measurement(m2, 'k2', [q(1, 0), q(0, 0), q(0, 1)], 1)
    _check_measurement(m3, 'k3', [q(1, 1), q(0, 2)], 1)