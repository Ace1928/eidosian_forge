import pytest
import pennylane.numpy as pnp
import pennylane as qml
def test_supports_tensor_observables(self, device_kwargs):
    """Tests that the device reports correctly whether it supports tensor observables."""
    device_kwargs['wires'] = 2
    dev = qml.device(**device_kwargs)
    if isinstance(dev, qml.devices.Device):
        pytest.skip('test is old interface specific.')
    cap = dev.capabilities()
    if 'supports_tensor_observables' not in cap:
        pytest.skip('No supports_tensor_observables capability specified by device.')

    @qml.qnode(dev)
    def circuit():
        """Model agnostic quantum function with tensor observable"""
        if cap['model'] == 'qubit':
            qml.X(0)
        else:
            qml.QuadX(wires=0)
        return qml.expval(qml.Identity(wires=0) @ qml.Identity(wires=1))
    if cap['supports_tensor_observables']:
        circuit()
    else:
        with pytest.raises(qml.QuantumFunctionError):
            circuit()