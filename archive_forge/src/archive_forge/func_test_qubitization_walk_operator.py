import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft import infra
from cirq_ft.algos.generic_select_test import get_1d_Ising_hamiltonian
from cirq_ft.algos.reflection_using_prepare_test import greedily_allocate_ancilla, keep
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@pytest.mark.slow
@pytest.mark.parametrize('num_sites,eps', [(4, 0.2), (3, 0.1)])
@allow_deprecated_cirq_ft_use_in_tests
def test_qubitization_walk_operator(num_sites: int, eps: float):
    ham = get_1d_Ising_hamiltonian(cirq.LineQubit.range(num_sites))
    ham_coeff = [abs(ps.coefficient.real) for ps in ham]
    qubitization_lambda = np.sum(ham_coeff)
    walk = walk_operator_for_pauli_hamiltonian(ham, eps)
    g = cirq_ft.testing.GateHelper(walk)
    context = cirq.DecompositionContext(cirq.ops.SimpleQubitManager())
    walk_circuit = cirq.Circuit(cirq.decompose(g.operation, keep=keep, on_stuck_raise=None, context=context))
    L_state = np.zeros(2 ** len(g.quregs['selection']))
    L_state[:len(ham_coeff)] = np.sqrt(ham_coeff / qubitization_lambda)
    greedy_mm = cirq.GreedyQubitManager('ancilla', maximize_reuse=True)
    walk_circuit = cirq.map_clean_and_borrowable_qubits(walk_circuit, qm=greedy_mm)
    assert len(walk_circuit.all_qubits()) < 23
    qubit_order = cirq.QubitOrder.explicit([*g.quregs['selection'], *g.quregs['target']], fallback=cirq.QubitOrder.DEFAULT)
    sim = cirq.Simulator(dtype=np.complex128)
    eigen_values, eigen_vectors = np.linalg.eigh(ham.matrix())
    for eig_idx, eig_val in enumerate(eigen_values):
        K_state = eigen_vectors[:, eig_idx].flatten()
        prep_L_K = cirq.Circuit(cirq.StatePreparationChannel(L_state, name='PREP_L').on(*g.quregs['selection']), cirq.StatePreparationChannel(K_state, name='PREP_K').on(*g.quregs['target']))
        L_K = sim.simulate(prep_L_K, qubit_order=qubit_order).final_state_vector
        prep_walk_circuit = prep_L_K + walk_circuit
        final_state = sim.simulate(prep_walk_circuit, qubit_order=qubit_order).final_state_vector
        final_state = final_state.reshape(len(L_K), -1).sum(axis=1)
        overlap = np.vdot(L_K, final_state)
        cirq.testing.assert_allclose_up_to_global_phase(overlap, eig_val / qubitization_lambda, atol=1e-06)