import pytest
import pennylane as qml
def test_tracker_initialization(self, device):
    """Tests a tracker instance is assigned at initialization."""
    dev = device(1)
    if isinstance(dev, qml.Device) and (not dev.capabilities().get('supports_tracker', False)):
        pytest.skip('Device does not support a tracker')
    assert isinstance(dev.tracker, qml.Tracker)