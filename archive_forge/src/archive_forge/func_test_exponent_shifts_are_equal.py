import itertools
import pytest
import numpy as np
import sympy
import cirq
def test_exponent_shifts_are_equal():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group((cirq.PauliInteractionGate(cirq.X, False, cirq.X, False, exponent=e) for e in [0.1, 0.1, 2.1, -1.9, 4.1]))
    eq.add_equality_group((cirq.PauliInteractionGate(cirq.X, True, cirq.X, False, exponent=e) for e in [0.1, 0.1, 2.1, -1.9, 4.1]))
    eq.add_equality_group((cirq.PauliInteractionGate(cirq.Y, False, cirq.Z, False, exponent=e) for e in [0.1, 0.1, 2.1, -1.9, 4.1]))
    eq.add_equality_group((cirq.PauliInteractionGate(cirq.Z, False, cirq.Y, True, exponent=e) for e in [0.1, 0.1, 2.1, -1.9, 4.1]))