import pytest
from flaky import flaky
import pennylane as qml
from pennylane import numpy as pnp  # Import from PennyLane to mirror the standard approach in demos
from pennylane.templates.layers import RandomLayers
def test_hermitian_expectation(self, device, tol, benchmark):
    """Test that arbitrary multi-mode Hermitian expectation values are correct"""
    n_wires = 2
    dev = device(n_wires)
    dev_def = qml.device('default.qubit')
    if dev.shots:
        pytest.skip('Device is in non-analytical mode.')
    if isinstance(dev, qml.Device) and 'Hermitian' not in dev.observables:
        pytest.skip('Device does not support the Hermitian observable.')
    if dev.name == 'default.qubit':
        pytest.skip('Device is default.qubit.')
    theta = 0.432
    phi = 0.123
    A_ = pnp.array([[-6, 2 + 1j, -3, -5 + 2j], [2 - 1j, 0, 2 - 1j, -5 + 4j], [-3, 2 + 1j, 0, -4 + 3j], [-5 - 2j, -5 - 4j, -4 - 3j, -6]], requires_grad=False)

    def circuit(theta, phi):
        qml.RX(theta, wires=[0])
        qml.RX(phi, wires=[1])
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.Hermitian(A_, wires=[0, 1]))
    qnode_def = qml.QNode(circuit, dev_def)
    qnode = qml.QNode(circuit, dev)
    grad_def = qml.grad(qnode_def, argnum=[0, 1])
    grad = qml.grad(qnode, argnum=[0, 1])

    def workload():
        return (qnode(theta, phi), qnode_def(theta, phi), grad(theta, phi), grad_def(theta, phi))
    qnode_res, qnode_def_res, grad_res, grad_def_res = benchmark(workload)
    assert pnp.allclose(qnode_res, qnode_def_res, atol=tol(dev.shots))
    assert pnp.allclose(grad_res, grad_def_res, atol=tol(dev.shots))