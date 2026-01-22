import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft import infra
from cirq_ft.infra.bit_tools import iter_bits
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_selected_majorana_fermion_gate_diagram():
    selection_bitsize, target_bitsize = (3, 5)
    gate = cirq_ft.SelectedMajoranaFermionGate(cirq_ft.SelectionRegister('selection', selection_bitsize, target_bitsize), target_gate=cirq.X)
    circuit = cirq.Circuit(gate.on_registers(**infra.get_named_qubits(gate.signature)))
    qubits = list((q for v in infra.get_named_qubits(gate.signature).values() for q in v))
    cirq.testing.assert_has_diagram(circuit, '\ncontrol: ──────@────\n               │\nselection0: ───In───\n               │\nselection1: ───In───\n               │\nselection2: ───In───\n               │\ntarget0: ──────ZX───\n               │\ntarget1: ──────ZX───\n               │\ntarget2: ──────ZX───\n               │\ntarget3: ──────ZX───\n               │\ntarget4: ──────ZX───\n', qubit_order=qubits)