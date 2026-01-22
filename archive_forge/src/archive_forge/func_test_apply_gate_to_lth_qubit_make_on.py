import cirq
import cirq_ft
import pytest
from cirq_ft import infra
from cirq_ft.infra.bit_tools import iter_bits
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_apply_gate_to_lth_qubit_make_on():
    gate = cirq_ft.ApplyGateToLthQubit(cirq_ft.SelectionRegister('selection', 3, 5), lambda n: cirq.Z if n & 1 else cirq.I, control_regs=cirq_ft.Signature.build(control=2))
    op = gate.on_registers(**infra.get_named_qubits(gate.signature))
    op2 = cirq_ft.ApplyGateToLthQubit.make_on(nth_gate=lambda n: cirq.Z if n & 1 else cirq.I, **infra.get_named_qubits(gate.signature))
    assert op.qubits == op2.qubits
    assert op.gate.selection_regs == op2.gate.selection_regs
    assert op.gate.control_regs == op2.gate.control_regs