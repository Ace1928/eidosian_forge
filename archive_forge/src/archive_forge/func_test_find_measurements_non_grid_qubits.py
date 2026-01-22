import numpy as np
import pytest
import cirq
import cirq_google
from cirq_google.api import v2
def test_find_measurements_non_grid_qubits():
    circuit = cirq.Circuit()
    circuit.append(cirq.measure(cirq.NamedQubit('a'), key='k'))
    with pytest.raises(ValueError, match='Expected GridQubits'):
        v2.find_measurements(circuit)