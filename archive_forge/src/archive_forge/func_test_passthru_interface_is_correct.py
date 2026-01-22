import pytest
import pennylane.numpy as pnp
import pennylane as qml
def test_passthru_interface_is_correct(self, device_kwargs):
    """Test that the capabilities dictionary defines a valid passthru interface, if not None."""
    device_kwargs['wires'] = 1
    dev = qml.device(**device_kwargs)
    if isinstance(dev, qml.devices.Device):
        pytest.skip('test is old interface specific.')
    cap = dev.capabilities()
    if 'passthru_interface' not in cap:
        pytest.skip('No passthru_interface capability specified by device.')
    interface = cap['passthru_interface']
    assert interface in ['tf', 'autograd', 'jax', 'torch']
    qfunc = qfunc_with_scalar_input(cap['model'])
    qnode = qml.QNode(qfunc, dev, interface=interface)
    if interface == 'tf':
        if TF_SUPPORT:
            x = tf.Variable(0.1)
            with tf.GradientTape() as tape:
                res = qnode(x)
                tape.gradient(res, [x])
        else:
            pytest.skip('Cannot import tensorflow.')
    if interface == 'autograd':
        x = pnp.array(0.1, requires_grad=True)
        g = qml.grad(qnode)
        g(x)
    if interface == 'jax':
        if JAX_SUPPORT:
            x = pnp.array(0.1, requires_grad=True)
            g = jax.grad(lambda a: qnode(a).reshape(()))
            g(x)
        else:
            pytest.skip('Cannot import jax')
    if interface == 'torch':
        if TORCH_SUPPORT:
            x = torch.tensor(0.1, requires_grad=True)
            res = qnode(x)
            res.backward()
            assert hasattr(x, 'grad')
        else:
            pytest.skip('Cannot import torch')