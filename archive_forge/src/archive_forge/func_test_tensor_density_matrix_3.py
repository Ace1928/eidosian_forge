import numpy as np
import cirq
import cirq.contrib.quimb as ccq
def test_tensor_density_matrix_3():
    qubits = cirq.LineQubit.range(10)
    circuit = cirq.testing.random_circuit(qubits=qubits, n_moments=10, op_density=0.8)
    rho1 = cirq.final_density_matrix(circuit, dtype=np.complex128)
    rho2 = ccq.tensor_density_matrix(circuit, qubits)
    np.testing.assert_allclose(rho1, rho2, atol=1e-08)