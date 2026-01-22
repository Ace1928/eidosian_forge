import itertools
from typing import Sequence, Tuple
import cirq
import cirq_ft
import pytest
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.infra.bit_tools import iter_bits
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_unary_iteration_loop():
    n_range, m_range = ((3, 5), (6, 8))
    selection_registers = [cirq_ft.SelectionRegister('n', 3, 5), cirq_ft.SelectionRegister('m', 3, 8)]
    selection = infra.get_named_qubits(selection_registers)
    target = {(n, m): cirq.q(f't({n}, {m})') for n in range(*n_range) for m in range(*m_range)}
    qm = cirq.GreedyQubitManager('ancilla', maximize_reuse=True)
    circuit = cirq.Circuit()
    i_ops = []
    for i_optree, i_ctrl, i in cirq_ft.unary_iteration(n_range[0], n_range[1], i_ops, [], selection['n'], qm):
        circuit.append(i_optree)
        j_ops = []
        for j_optree, j_ctrl, j in cirq_ft.unary_iteration(m_range[0], m_range[1], j_ops, [i_ctrl], selection['m'], qm):
            circuit.append(j_optree)
            circuit.append(cirq.CNOT(j_ctrl, target[i, j]))
        circuit.append(j_ops)
    circuit.append(i_ops)
    all_qubits = sorted(circuit.all_qubits())
    i_len, j_len = (3, 3)
    for i, j in itertools.product(range(*n_range), range(*m_range)):
        qubit_vals = {x: 0 for x in all_qubits}
        qubit_vals.update(zip(selection['n'], iter_bits(i, i_len)))
        qubit_vals.update(zip(selection['m'], iter_bits(j, j_len)))
        initial_state = [qubit_vals[x] for x in all_qubits]
        qubit_vals[target[i, j]] = 1
        final_state = [qubit_vals[x] for x in all_qubits]
        cirq_ft.testing.assert_circuit_inp_out_cirqsim(circuit, all_qubits, initial_state, final_state)