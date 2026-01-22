import itertools
import pytest
import numpy as np
import sympy
import cirq
def test_eq_ne_hash():
    q0, q1, q2, q3 = _make_qubits(4)
    eq = cirq.testing.EqualsTester()
    ps1 = cirq.X(q0) * cirq.Y(q1) * cirq.Z(q2)
    ps2 = cirq.X(q0) * cirq.Y(q1) * cirq.X(q2)
    eq.make_equality_group(lambda: cirq.PauliStringPhasor(cirq.PauliString(), exponent_neg=0.5), lambda: cirq.PauliStringPhasor(cirq.PauliString(), exponent_neg=-1.5), lambda: cirq.PauliStringPhasor(cirq.PauliString(), exponent_neg=2.5))
    eq.make_equality_group(lambda: cirq.PauliStringPhasor(-cirq.PauliString(), exponent_neg=-0.5))
    eq.add_equality_group(cirq.PauliStringPhasor(ps1), cirq.PauliStringPhasor(ps1, exponent_neg=1))
    eq.add_equality_group(cirq.PauliStringPhasor(-ps1, exponent_neg=1))
    eq.add_equality_group(cirq.PauliStringPhasor(ps2), cirq.PauliStringPhasor(ps2, exponent_neg=1))
    eq.add_equality_group(cirq.PauliStringPhasor(-ps2, exponent_neg=1))
    eq.add_equality_group(cirq.PauliStringPhasor(ps2, exponent_neg=0.5))
    eq.add_equality_group(cirq.PauliStringPhasor(-ps2, exponent_neg=-0.5))
    eq.add_equality_group(cirq.PauliStringPhasor(ps1, exponent_neg=sympy.Symbol('a')))
    eq.add_equality_group(cirq.PauliStringPhasor(ps1, qubits=[q0, q1, q2, q3]))