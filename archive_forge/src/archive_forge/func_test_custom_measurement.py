import pytest
from flaky import flaky
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane import numpy as np
from pennylane.measurements import (
def test_custom_measurement(self, device):
    """Test the execution of a custom measurement."""
    dev = device(2)
    _skip_test_for_braket(dev)

    class MyMeasurement(MeasurementTransform):
        """Dummy measurement transform."""

        def process(self, tape, device):
            return 1
    if isinstance(dev, qml.devices.Device):
        tape = qml.tape.QuantumScript([], [MyMeasurement()])
        try:
            dev.preprocess()[0]((tape,))
        except qml.DeviceError:
            pytest.xfail('Device does not support custom measurement transforms.')

    @qml.qnode(dev)
    def circuit():
        qml.X(0)
        return MyMeasurement()
    assert circuit() == 1