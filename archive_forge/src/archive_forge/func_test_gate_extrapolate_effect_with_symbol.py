import itertools
import pytest
import numpy as np
import sympy
import cirq
def test_gate_extrapolate_effect_with_symbol():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq.PauliStringPhasorGate(dps_empty, exponent_neg=sympy.Symbol('a')), cirq.PauliStringPhasorGate(dps_empty) ** sympy.Symbol('a'))
    eq.add_equality_group(cirq.PauliStringPhasorGate(dps_empty) ** sympy.Symbol('b'))
    eq.add_equality_group(cirq.PauliStringPhasorGate(dps_empty, exponent_neg=0.5) ** sympy.Symbol('b'))
    eq.add_equality_group(cirq.PauliStringPhasorGate(dps_empty, exponent_neg=sympy.Symbol('a')) ** 0.5)
    eq.add_equality_group(cirq.PauliStringPhasorGate(dps_empty, exponent_neg=sympy.Symbol('a')) ** sympy.Symbol('b'))