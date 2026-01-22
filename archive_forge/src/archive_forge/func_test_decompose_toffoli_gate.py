import cirq
import cirq_ionq as ionq
import pytest
import sympy
def test_decompose_toffoli_gate():
    """Decompose result should reflect all-to-all connectivity"""
    circuit = cirq.Circuit(cirq.TOFFOLI(*cirq.LineQubit.range(3)))
    decomposed_circuit = cirq.optimize_for_target_gateset(circuit, gateset=ionq_target_gateset, ignore_failures=False)
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(circuit, decomposed_circuit, atol=1e-08)
    assert ionq_target_gateset.validate(decomposed_circuit)
    cirq.testing.assert_has_diagram(decomposed_circuit, '\n0: ──────────────────@──────────────────@───@───T──────@───\n                     │                  │   │          │\n1: ───────@──────────┼───────@───T──────┼───X───T^-1───X───\n          │          │       │          │\n2: ───H───X───T^-1───X───T───X───T^-1───X───T───H──────────\n')