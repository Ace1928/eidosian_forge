import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft import infra
from cirq_ft.algos.generic_select_test import get_1d_Ising_hamiltonian
from cirq_ft.algos.reflection_using_prepare_test import greedily_allocate_ancilla, keep
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_qubitization_walk_operator_diagrams():
    num_sites, eps = (4, 0.1)
    walk = get_walk_operator_for_1d_Ising_model(num_sites, eps)
    qu_regs = infra.get_named_qubits(walk.signature)
    walk_op = walk.on_registers(**qu_regs)
    circuit = cirq.Circuit(cirq.decompose_once(walk_op))
    cirq.testing.assert_has_diagram(circuit, '\nselection0: ───In──────────────R_L───\n               │               │\nselection1: ───In──────────────R_L───\n               │               │\nselection2: ───In──────────────R_L───\n               │\ntarget0: ──────GenericSelect─────────\n               │\ntarget1: ──────GenericSelect─────────\n               │\ntarget2: ──────GenericSelect─────────\n               │\ntarget3: ──────GenericSelect─────────\n')
    walk_squared_op = walk.with_power(2).on_registers(**qu_regs)
    circuit = cirq.Circuit(cirq.decompose_once(walk_squared_op))
    cirq.testing.assert_has_diagram(circuit, '\nselection0: ───In──────────────R_L───In──────────────R_L───\n               │               │     │               │\nselection1: ───In──────────────R_L───In──────────────R_L───\n               │               │     │               │\nselection2: ───In──────────────R_L───In──────────────R_L───\n               │                     │\ntarget0: ──────GenericSelect─────────GenericSelect─────────\n               │                     │\ntarget1: ──────GenericSelect─────────GenericSelect─────────\n               │                     │\ntarget2: ──────GenericSelect─────────GenericSelect─────────\n               │                     │\ntarget3: ──────GenericSelect─────────GenericSelect─────────\n')
    controlled_walk_op = walk.controlled().on_registers(**qu_regs, control=cirq.q('control'))
    circuit = cirq.Circuit(cirq.decompose_once(controlled_walk_op))
    cirq.testing.assert_has_diagram(circuit, '\ncontrol: ──────@───────────────@─────\n               │               │\nselection0: ───In──────────────R_L───\n               │               │\nselection1: ───In──────────────R_L───\n               │               │\nselection2: ───In──────────────R_L───\n               │\ntarget0: ──────GenericSelect─────────\n               │\ntarget1: ──────GenericSelect─────────\n               │\ntarget2: ──────GenericSelect─────────\n               │\ntarget3: ──────GenericSelect─────────\n')
    gateset_to_keep = cirq.Gateset(cirq_ft.GenericSelect, cirq_ft.StatePreparationAliasSampling, cirq_ft.MultiControlPauli, cirq.X)

    def keep(op):
        ret = op in gateset_to_keep
        if op.gate is not None and isinstance(op.gate, cirq.ops.raw_types._InverseCompositeGate):
            ret |= op.gate._original in gateset_to_keep
        return ret
    circuit = cirq.Circuit(cirq.decompose(controlled_walk_op, keep=keep, on_stuck_raise=None))
    circuit = greedily_allocate_ancilla(circuit)
    cirq.testing.assert_has_diagram(circuit, '\nancilla_0: ────────────────────sigma_mu───────────────────────────────sigma_mu────────────────────────\n                               │                                      │\nancilla_1: ────────────────────alt────────────────────────────────────alt─────────────────────────────\n                               │                                      │\nancilla_2: ────────────────────alt────────────────────────────────────alt─────────────────────────────\n                               │                                      │\nancilla_3: ────────────────────alt────────────────────────────────────alt─────────────────────────────\n                               │                                      │\nancilla_4: ────────────────────keep───────────────────────────────────keep────────────────────────────\n                               │                                      │\nancilla_5: ────────────────────less_than_equal────────────────────────less_than_equal─────────────────\n                               │                                      │\ncontrol: ──────@───────────────┼───────────────────────────────Z──────┼───────────────────────────────\n               │               │                               │      │\nselection0: ───In──────────────StatePreparationAliasSampling───@(0)───StatePreparationAliasSampling───\n               │               │                               │      │\nselection1: ───In──────────────selection───────────────────────@(0)───selection───────────────────────\n               │               │                               │      │\nselection2: ───In──────────────selection^-1────────────────────@(0)───selection───────────────────────\n               │\ntarget0: ──────GenericSelect──────────────────────────────────────────────────────────────────────────\n               │\ntarget1: ──────GenericSelect──────────────────────────────────────────────────────────────────────────\n               │\ntarget2: ──────GenericSelect──────────────────────────────────────────────────────────────────────────\n               │\ntarget3: ──────GenericSelect──────────────────────────────────────────────────────────────────────────')