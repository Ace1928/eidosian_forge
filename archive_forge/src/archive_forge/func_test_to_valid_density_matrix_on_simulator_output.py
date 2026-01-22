import itertools
import numpy as np
import pytest
import cirq
import cirq.testing
@pytest.mark.parametrize('seed', [17, 35, 48])
@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
@pytest.mark.parametrize('split', [False, True])
def test_to_valid_density_matrix_on_simulator_output(seed, dtype, split):
    circuit = cirq.testing.random_circuit(qubits=5, n_moments=20, op_density=0.9, random_state=seed)
    simulator = cirq.DensityMatrixSimulator(split_untangled_states=split, dtype=dtype)
    result = simulator.simulate(circuit)
    _ = cirq.to_valid_density_matrix(result.final_density_matrix, num_qubits=5, atol=1e-06)