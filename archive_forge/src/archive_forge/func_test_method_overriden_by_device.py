import pytest
from flaky import flaky
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane import numpy as np
from pennylane.measurements import (
def test_method_overriden_by_device(self, device):
    """Test that the device can override a measurement process."""
    dev = device(2)
    if isinstance(dev, qml.devices.Device):
        pytest.skip('test specific to old device interface.')
    _skip_test_for_braket(dev)
    if dev.shots is None:
        pytest.skip('The number of shots has to be explicitly set on the device when using sample-based measurements.')

    @qml.qnode(dev)
    def circuit():
        qml.X(0)
        return qml.classical_shadow(wires=0)
    circuit.device.measurement_map[ClassicalShadowMP] = 'test_method'
    circuit.device.test_method = lambda tape: 2
    assert circuit() == 2