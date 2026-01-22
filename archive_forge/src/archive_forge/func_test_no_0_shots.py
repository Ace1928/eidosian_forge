import pytest
import pennylane.numpy as pnp
import pennylane as qml
def test_no_0_shots(self, device_kwargs):
    """Test that non-analytic devices cannot accept 0 shots."""
    device_kwargs['wires'] = 2
    device_kwargs['shots'] = 0
    with pytest.raises(Exception):
        dev = qml.device(**device_kwargs)
        if isinstance(dev, qml.devices.Device):
            pytest.skip('test is old interface specific.')