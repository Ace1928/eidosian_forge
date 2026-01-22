import dataclasses
import pytest
import numpy as np
import sympy
import cirq
from cirq.transformers.eject_z import _is_swaplike
def test_removes_zs():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    assert_removes_all_z_gates(cirq.Circuit(cirq.Z(a), cirq.measure(a)))
    assert_removes_all_z_gates(cirq.Circuit(cirq.Z(a), cirq.measure(a, b)))
    assert_removes_all_z_gates(cirq.Circuit(cirq.Z(a), cirq.Z(a), cirq.measure(a)))
    assert_removes_all_z_gates(cirq.Circuit(cirq.Z(a), cirq.measure(a, key='k')))
    assert_removes_all_z_gates(cirq.Circuit(cirq.Z(a), cirq.X(a), cirq.measure(a)))
    assert_removes_all_z_gates(cirq.Circuit(cirq.Z(a), cirq.X(a), cirq.X(a), cirq.measure(a)))
    assert_removes_all_z_gates(cirq.Circuit(cirq.Z(a), cirq.Z(b), cirq.CZ(a, b), cirq.CZ(a, b), cirq.measure(a, b)))
    assert_removes_all_z_gates(cirq.Circuit(cirq.PhasedXZGate(axis_phase_exponent=0, x_exponent=0, z_exponent=1).on(a), cirq.measure(a)))
    assert_removes_all_z_gates(cirq.Circuit(cirq.Z(a) ** sympy.Symbol('a'), cirq.Z(b) ** (sympy.Symbol('a') + 1), cirq.CZ(a, b), cirq.CZ(a, b), cirq.measure(a, b)), eject_parameterized=True)