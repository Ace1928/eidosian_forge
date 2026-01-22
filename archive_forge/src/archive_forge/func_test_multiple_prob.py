import pytest
import pennylane as qml
from pennylane import numpy as np  # Import from PennyLane to mirror the standard approach in demos
def test_multiple_prob(self, device):
    """Return multiple probs."""
    n_wires = 2
    dev = device(n_wires)

    def circuit(x):
        qubit_ansatz(x)
        return (qml.probs(op=qml.Z(0)), qml.probs(op=qml.Y(1)))
    qnode = qml.QNode(circuit, dev, diff_method=None)
    res = qnode(0.5)
    assert isinstance(res, tuple)
    assert len(res) == 2
    assert isinstance(res[0], np.ndarray)
    assert res[0].shape == (2 ** 1,)
    assert isinstance(res[1], np.ndarray)
    assert res[1].shape == (2 ** 1,)