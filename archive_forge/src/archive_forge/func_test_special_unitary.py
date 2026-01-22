from cmath import exp
from math import cos, sin, sqrt
import pytest
import numpy as np
from scipy.linalg import block_diag
from flaky import flaky
import pennylane as qml
@pytest.mark.parametrize('theta_', [np.array([0.4, -0.1, 0.2]), np.ones(15) / 3])
def test_special_unitary(self, device, init_state, theta_, tol, skip_if, benchmark):
    """Test SpecialUnitary gate."""
    n_wires = int(np.log(len(theta_) + 1) / np.log(4))
    dev = device(n_wires)
    if isinstance(dev, qml.Device):
        if 'SpecialUnitary' not in dev.operations:
            pytest.skip('Skipped because device does not support SpecialUnitary.')
        skip_if(dev, {'returns_probs': False})
    rnd_state = init_state(n_wires)

    @qml.qnode(dev)
    def circuit():
        qml.StatePrep(rnd_state, wires=range(n_wires))
        qml.SpecialUnitary(theta_, wires=list(range(n_wires)))
        return qml.probs(wires=range(n_wires))
    res = benchmark(circuit)
    basis_fn = qml.ops.qubit.special_unitary.pauli_basis_matrices
    basis = basis_fn(n_wires)
    mat = qml.math.expm(1j * np.tensordot(theta_, basis, axes=[[0], [0]]))
    expected = np.abs(mat @ rnd_state) ** 2
    assert np.allclose(res, expected, atol=tol(dev.shots))