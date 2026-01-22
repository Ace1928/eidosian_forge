from cmath import exp
from math import cos, sin, sqrt
import pytest
import numpy as np
from scipy.linalg import block_diag
from flaky import flaky
import pennylane as qml
@pytest.mark.parametrize('param', [0.5432, -0.232])
@pytest.mark.parametrize('op,func', two_qubit_param)
def test_two_qubit_parameters(self, device, init_state, op, func, param, tol, skip_if, benchmark):
    """Test parametrized two qubit gates taking a single scalar argument."""
    n_wires = 2
    dev = device(n_wires)
    if isinstance(dev, qml.Device):
        skip_if(dev, {'returns_probs': False})
    rnd_state = init_state(n_wires)

    @qml.qnode(dev)
    def circuit():
        qml.StatePrep(rnd_state, wires=range(n_wires))
        op(param, wires=range(n_wires))
        return qml.probs(wires=range(n_wires))
    res = benchmark(circuit)
    expected = np.abs(func(param) @ rnd_state) ** 2
    assert np.allclose(res, expected, atol=tol(dev.shots))