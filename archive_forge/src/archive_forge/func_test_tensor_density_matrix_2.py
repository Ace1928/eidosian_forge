import numpy as np
import cirq
import cirq.contrib.quimb as ccq
def test_tensor_density_matrix_2():
    q = cirq.LineQubit.range(2)
    rs = np.random.RandomState(52)
    for _ in range(10):
        g = cirq.MatrixGate(cirq.testing.random_unitary(dim=2 ** len(q), random_state=rs))
        c = cirq.Circuit(g.on(*q))
        rho1 = cirq.final_density_matrix(c, dtype=np.complex128)
        rho2 = ccq.tensor_density_matrix(c, q)
        np.testing.assert_allclose(rho1, rho2, atol=1e-08)