from typing import cast, Iterable
import dataclasses
import numpy as np
import pytest
import sympy
import cirq
def test_crosses_czs():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    z = sympy.Symbol('z')
    assert_optimizes(before=quick_circuit([cirq.PhasedXPowGate(phase_exponent=0.25).on(a)], [cirq.CZ(a, b)]), expected=quick_circuit([cirq.Z(b)], [cirq.CZ(a, b)], [cirq.PhasedXPowGate(phase_exponent=0.25).on(a)]))
    assert_optimizes(before=quick_circuit([cirq.PhasedXPowGate(phase_exponent=0.125).on(a)], [cirq.CZ(b, a)]), expected=quick_circuit([cirq.Z(b)], [cirq.CZ(a, b)], [cirq.PhasedXPowGate(phase_exponent=0.125).on(a)]))
    assert_optimizes(before=quick_circuit([cirq.PhasedXPowGate(phase_exponent=x).on(a)], [cirq.CZ(b, a)]), expected=quick_circuit([cirq.Z(b)], [cirq.CZ(a, b)], [cirq.PhasedXPowGate(phase_exponent=x).on(a)]), eject_parameterized=True)
    assert_optimizes(before=quick_circuit([cirq.X(a)], [cirq.CZ(a, b) ** 0.25]), expected=quick_circuit([cirq.Z(b) ** 0.25], [cirq.CZ(a, b) ** (-0.25)], [cirq.X(a)]))
    assert_optimizes(before=quick_circuit([cirq.X(a)], [cirq.CZ(a, b) ** x]), expected=quick_circuit([cirq.Z(b) ** x], [cirq.CZ(a, b) ** (-x)], [cirq.X(a)]), eject_parameterized=True)
    assert_optimizes(before=quick_circuit([cirq.PhasedXPowGate(phase_exponent=0.125).on(a)], [cirq.PhasedXPowGate(phase_exponent=0.375).on(b)], [cirq.CZ(a, b) ** 0.25]), expected=quick_circuit([cirq.CZ(a, b) ** 0.25], [cirq.PhasedXPowGate(phase_exponent=0.5).on(b), cirq.PhasedXPowGate(phase_exponent=0.25).on(a)]))
    assert_optimizes(before=quick_circuit([cirq.PhasedXPowGate(phase_exponent=x).on(a)], [cirq.PhasedXPowGate(phase_exponent=y).on(b)], [cirq.CZ(a, b) ** z]), expected=quick_circuit([cirq.CZ(a, b) ** z], [cirq.PhasedXPowGate(phase_exponent=y + z / 2).on(b), cirq.PhasedXPowGate(phase_exponent=x + z / 2).on(a)]), eject_parameterized=True)