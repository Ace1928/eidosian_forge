import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft import infra
from cirq_ft.infra.bit_tools import iter_bits
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_selected_majorana_fermion_gate_make_on():
    selection_bitsize, target_bitsize = (3, 5)
    gate = cirq_ft.SelectedMajoranaFermionGate(cirq_ft.SelectionRegister('selection', selection_bitsize, target_bitsize), target_gate=cirq.X)
    op = gate.on_registers(**infra.get_named_qubits(gate.signature))
    op2 = cirq_ft.SelectedMajoranaFermionGate.make_on(target_gate=cirq.X, **infra.get_named_qubits(gate.signature))
    assert op == op2