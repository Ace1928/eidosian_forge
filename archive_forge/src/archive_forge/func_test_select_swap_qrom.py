import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft import infra
from cirq_ft.infra.bit_tools import iter_bits
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@pytest.mark.parametrize('data,block_size', [pytest.param(data, block_size, id=f'{block_size}-data{didx}', marks=pytest.mark.slow if block_size == 3 and didx == 1 else ()) for didx, data in enumerate([[[1, 2, 3, 4, 5]], [[1, 2, 3], [3, 2, 1]]]) for block_size in [None, 1, 2, 3]])
@allow_deprecated_cirq_ft_use_in_tests
def test_select_swap_qrom(data, block_size):
    qrom = cirq_ft.SelectSwapQROM(*data, block_size=block_size)
    qubit_regs = infra.get_named_qubits(qrom.signature)
    selection = qubit_regs['selection']
    selection_q, selection_r = (selection[:qrom.selection_q], selection[qrom.selection_q:])
    targets = [qubit_regs[f'target{i}'] for i in range(len(data))]
    greedy_mm = cirq.GreedyQubitManager(prefix='_a', maximize_reuse=True)
    context = cirq.DecompositionContext(greedy_mm)
    qrom_circuit = cirq.Circuit(cirq.decompose(qrom.on_registers(**qubit_regs), context=context))
    dirty_target_ancilla = [q for q in qrom_circuit.all_qubits() if isinstance(q, cirq.ops.BorrowableQubit)]
    circuit = cirq.Circuit(cirq.H.on_each(*dirty_target_ancilla), cirq.T.on_each(*dirty_target_ancilla), *qrom_circuit, (cirq.T ** (-1)).on_each(*dirty_target_ancilla), cirq.H.on_each(*dirty_target_ancilla))
    all_qubits = sorted(circuit.all_qubits())
    for selection_integer in range(qrom.selection_registers[0].iteration_length):
        svals_q = list(iter_bits(selection_integer // qrom.block_size, len(selection_q)))
        svals_r = list(iter_bits(selection_integer % qrom.block_size, len(selection_r)))
        qubit_vals = {x: 0 for x in all_qubits}
        qubit_vals.update({s: sval for s, sval in zip(selection_q, svals_q)})
        qubit_vals.update({s: sval for s, sval in zip(selection_r, svals_r)})
        dvals = np.random.randint(2, size=len(dirty_target_ancilla))
        qubit_vals.update({d: dval for d, dval in zip(dirty_target_ancilla, dvals)})
        initial_state = [qubit_vals[x] for x in all_qubits]
        for target, d in zip(targets, data):
            for q, b in zip(target, iter_bits(d[selection_integer], len(target))):
                qubit_vals[q] = b
        final_state = [qubit_vals[x] for x in all_qubits]
        cirq_ft.testing.assert_circuit_inp_out_cirqsim(circuit, all_qubits, initial_state, final_state)