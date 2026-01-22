from typing import List, Sequence
import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft import infra
from cirq_ft.infra.bit_tools import iter_bits
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_select_application_to_eigenstates():
    sim = cirq.Simulator(dtype=np.complex128)
    num_sites = 3
    target_bitsize = num_sites
    num_select_unitaries = 2 * num_sites
    selection_bitsize = int(np.ceil(np.log2(num_select_unitaries)))
    all_qubits = cirq.LineQubit.range(2 * selection_bitsize + target_bitsize + 1)
    control, selection, target = (all_qubits[0], all_qubits[1:2 * selection_bitsize:2], all_qubits[2 * selection_bitsize + 1:])
    ham = get_1d_Ising_hamiltonian(target, 1, 1)
    dense_pauli_string_hamiltonian = [tt.dense(target) for tt in ham]
    op = cirq_ft.GenericSelect(selection_bitsize=selection_bitsize, target_bitsize=target_bitsize, select_unitaries=dense_pauli_string_hamiltonian, control_val=1).on(control, *selection, *target)
    select_circuit = cirq.Circuit(cirq.decompose(op))
    all_qubits = select_circuit.all_qubits()
    coeffs = get_1d_Ising_lcu_coeffs(num_sites, 1, 1)
    prep_circuit = _fake_prepare(np.sqrt(coeffs), selection)
    turn_on_control = cirq.Circuit(cirq.X.on(control))
    ising_eigs, ising_wfns = np.linalg.eigh(ham.matrix())
    qubitization_lambda = sum((xx.coefficient.real for xx in dense_pauli_string_hamiltonian))
    for iw_idx, ie in enumerate(ising_eigs):
        eigenstate_prep = cirq.Circuit()
        eigenstate_prep.append(cirq.StatePreparationChannel(ising_wfns[:, iw_idx].flatten()).on(*target))
        input_circuit = turn_on_control + prep_circuit + eigenstate_prep
        input_vec = sim.simulate(input_circuit, qubit_order=all_qubits).final_state_vector
        final_circuit = input_circuit + select_circuit
        out_vec = sim.simulate(final_circuit, qubit_order=all_qubits).final_state_vector
        np.testing.assert_allclose(np.vdot(input_vec, out_vec), ie / qubitization_lambda, atol=1e-08)