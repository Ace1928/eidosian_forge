import itertools
import pytest
import numpy as np
import sympy
import cirq
def test_extrapolate_effect_with_symbol():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq.PauliStringPhasor(cirq.PauliString({}), exponent_neg=sympy.Symbol('a')), cirq.PauliStringPhasor(cirq.PauliString({})) ** sympy.Symbol('a'))
    eq.add_equality_group(cirq.PauliStringPhasor(cirq.PauliString({})) ** sympy.Symbol('b'))
    eq.add_equality_group(cirq.PauliStringPhasor(cirq.PauliString({}), exponent_neg=0.5) ** sympy.Symbol('b'))
    eq.add_equality_group(cirq.PauliStringPhasor(cirq.PauliString({}), exponent_neg=sympy.Symbol('a')) ** 0.5)
    eq.add_equality_group(cirq.PauliStringPhasor(cirq.PauliString({}), exponent_neg=sympy.Symbol('a')) ** sympy.Symbol('b'))