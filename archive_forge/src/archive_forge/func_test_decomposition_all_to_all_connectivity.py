import cirq
import cirq_ionq as ionq
import pytest
import sympy
def test_decomposition_all_to_all_connectivity():
    """This function only accepts 3 qubits as input"""
    with pytest.raises(ValueError):
        decompose_result = ionq.decompose_all_to_all_connect_ccz_gate(cirq.CCZ, cirq.LineQubit.range(4))
    decompose_result = ionq.decompose_all_to_all_connect_ccz_gate(cirq.CCZ, cirq.LineQubit.range(3))
    cirq.testing.assert_has_diagram(cirq.Circuit(decompose_result), '\n0: ──────────────@──────────────────@───@───T──────@───\n                 │                  │   │          │\n1: ───@──────────┼───────@───T──────┼───X───T^-1───X───\n      │          │       │          │\n2: ───X───T^-1───X───T───X───T^-1───X───T──────────────\n')