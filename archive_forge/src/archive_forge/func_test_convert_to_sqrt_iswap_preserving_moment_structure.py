from typing import Optional
import cirq
import pytest
import sympy
import numpy as np
def test_convert_to_sqrt_iswap_preserving_moment_structure():
    q = cirq.LineQubit.range(5)
    op = lambda q0, q1: cirq.H(q1).controlled_by(q0)
    c_orig = cirq.Circuit(cirq.Moment(cirq.X(q[2])), cirq.Moment(op(q[0], q[1]), op(q[2], q[3])), cirq.Moment(op(q[2], q[1]), op(q[4], q[3])), cirq.Moment(op(q[1], q[2]), op(q[3], q[4])), cirq.Moment(op(q[3], q[2]), op(q[1], q[0])), cirq.measure(*q[:2], key='m'), cirq.X(q[2]).with_classical_controls('m'), cirq.CZ(*q[3:]).with_classical_controls('m'))
    c_new = cirq.optimize_for_target_gateset(c_orig, gateset=cirq.SqrtIswapTargetGateset(), ignore_failures=True)
    assert c_orig[-2:] == c_new[-2:]
    c_orig, c_new = (c_orig[:-2], c_new[:-2])
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(c_orig, c_new, atol=1e-06)
    assert all((all_gates_of_type(m, cirq.Gateset(cirq.PhasedXZGate)) or all_gates_of_type(m, cirq.Gateset(cirq.SQRT_ISWAP)) for m in c_new))
    c_new = cirq.optimize_for_target_gateset(c_orig, gateset=cirq.SqrtIswapTargetGateset(use_sqrt_iswap_inv=True), ignore_failures=False)
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(c_orig, c_new, atol=1e-06)
    assert all((all_gates_of_type(m, cirq.Gateset(cirq.PhasedXZGate)) or all_gates_of_type(m, cirq.Gateset(cirq.SQRT_ISWAP_INV)) for m in c_new))