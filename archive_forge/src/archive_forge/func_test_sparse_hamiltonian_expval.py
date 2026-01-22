import pytest
from flaky import flaky
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane import numpy as np
from pennylane.measurements import (
def test_sparse_hamiltonian_expval(self, device, tol):
    """Test that expectation values of sparse Hamiltonians are properly calculated."""
    n_wires = 4
    dev = device(n_wires)
    if isinstance(dev, qml.Device):
        if 'SparseHamiltonian' not in dev.observables:
            pytest.skip('Skipped because device does not support the SparseHamiltonian observable.')
    if dev.shots:
        pytest.skip('SparseHamiltonian only supported in analytic mode')
    h_row = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    h_col = np.array([15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
    h_data = np.array([-1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1], dtype=np.complex128)
    h = csr_matrix((h_data, (h_row, h_col)), shape=(16, 16))

    @qml.qnode(dev, diff_method='parameter-shift')
    def result():
        qml.X(0)
        qml.X(2)
        qml.SingleExcitation(0.1, wires=[0, 1])
        qml.SingleExcitation(0.2, wires=[2, 3])
        qml.SingleExcitation(0.3, wires=[1, 2])
        return qml.expval(qml.SparseHamiltonian(h, wires=[0, 1, 2, 3]))
    res = result()
    exp_res = 0.019833838076209875
    assert np.allclose(res, exp_res, atol=tol(False))