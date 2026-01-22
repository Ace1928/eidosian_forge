import pytest
import sympy
import numpy as np
import cirq
import cirq_pasqal
from cirq_pasqal import PasqalDevice, PasqalVirtualDevice
from cirq_pasqal import TwoDQubit, ThreeDQubit
def test_minimal_distance():
    dev = square_virtual_device(control_r=1.0, num_qubits=1)
    with pytest.raises(ValueError, match='Two qubits to compute a minimal distance.'):
        dev.minimal_distance()
    dev = square_virtual_device(control_r=1.0, num_qubits=2)
    assert np.isclose(dev.minimal_distance(), 1.0)