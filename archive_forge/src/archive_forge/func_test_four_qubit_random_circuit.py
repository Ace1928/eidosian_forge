import pytest
from flaky import flaky
import pennylane as qml
from pennylane import numpy as pnp  # Import from PennyLane to mirror the standard approach in demos
from pennylane.templates.layers import RandomLayers
def test_four_qubit_random_circuit(self, device, tol):
    """Compare a four-qubit random circuit with lots of different gates to default.qubit"""
    n_wires = 4
    dev = device(n_wires)
    dev_def = qml.device('default.qubit')
    if dev.name == dev_def.name:
        pytest.skip('Device is default.qubit.')
    if dev.shots:
        pytest.skip('Device is in non-analytical mode.')
    gates = [qml.X(0), qml.Y(1), qml.Z(2), qml.S(wires=3), qml.T(wires=0), qml.RX(2.3, wires=1), qml.RY(1.3, wires=2), qml.RZ(3.3, wires=3), qml.Hadamard(wires=0), qml.Rot(0.1, 0.2, 0.3, wires=1), qml.CRot(0.1, 0.2, 0.3, wires=[2, 3]), qml.Toffoli(wires=[0, 1, 2]), qml.SWAP(wires=[1, 2]), qml.CSWAP(wires=[1, 2, 3]), qml.U1(1.0, wires=0), qml.U2(1.0, 2.0, wires=2), qml.U3(1.0, 2.0, 3.0, wires=3), qml.CRX(0.1, wires=[1, 2]), qml.CRY(0.2, wires=[2, 3]), qml.CRZ(0.3, wires=[3, 1])]
    layers = 3
    rng = pnp.random.default_rng(1967)
    gates_per_layers = [rng.permutation(gates).numpy() for _ in range(layers)]

    def circuit():
        """4-qubit circuit with layers of randomly selected gates and random connections for
            multi-qubit gates."""
        for gates in gates_per_layers:
            for gate in gates:
                qml.apply(gate)
        return qml.expval(qml.Z(0))
    qnode_def = qml.QNode(circuit, dev_def)
    qnode = qml.QNode(circuit, dev)
    assert pnp.allclose(qnode(), qnode_def(), atol=tol(dev.shots))