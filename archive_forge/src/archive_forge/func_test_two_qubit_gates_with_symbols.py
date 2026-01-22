from typing import Optional
import cirq
import pytest
import sympy
import numpy as np
@pytest.mark.parametrize('gate', [cirq.CNotPowGate(exponent=sympy.Symbol('t')), cirq.PhasedFSimGate(theta=sympy.Symbol('t'), chi=sympy.Symbol('t'), phi=sympy.Symbol('t'))])
@pytest.mark.parametrize('use_sqrt_iswap_inv', [True, False])
def test_two_qubit_gates_with_symbols(gate: cirq.Gate, use_sqrt_iswap_inv: bool):
    c_orig = cirq.Circuit(gate(*cirq.LineQubit.range(2)))
    c_new = cirq.optimize_for_target_gateset(c_orig, gateset=cirq.SqrtIswapTargetGateset(use_sqrt_iswap_inv=use_sqrt_iswap_inv, additional_gates=[cirq.XPowGate, cirq.YPowGate, cirq.ZPowGate]), ignore_failures=False)
    sqrt_iswap_gate = cirq.SQRT_ISWAP_INV if use_sqrt_iswap_inv else cirq.SQRT_ISWAP
    for op in c_new.all_operations():
        if cirq.num_qubits(op) == 2:
            assert op.gate == sqrt_iswap_gate
    for val in np.linspace(0, 2 * np.pi, 10):
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(cirq.resolve_parameters(c_orig, {'t': val}), cirq.resolve_parameters(c_new, {'t': val}), atol=1e-06)