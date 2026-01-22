import pytest
import pennylane as qml
from pennylane import numpy as np
def make_simple_circuit_expval(device, wires):
    """Factory for a qnode returning expvals."""
    n_wires = len(wires)

    @qml.qnode(device)
    def circuit():
        qml.RX(0.5, wires=wires[0 % n_wires])
        qml.RY(2.0, wires=wires[1 % n_wires])
        if n_wires > 1:
            qml.CNOT(wires=[wires[0], wires[1]])
        return [qml.expval(qml.Z(w)) for w in wires]
    return circuit