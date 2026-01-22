import numpy as np
import pytest
import sympy
import cirq
def test_xx_eq():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq.XX, cirq.XXPowGate(), cirq.XXPowGate(exponent=1, global_shift=0), cirq.XXPowGate(exponent=3, global_shift=0))
    eq.add_equality_group(cirq.XX ** 0.5, cirq.XX ** 2.5, cirq.XX ** 4.5)
    eq.add_equality_group(cirq.XX ** 0.25, cirq.XX ** 2.25, cirq.XX ** (-1.75))
    iXX = cirq.XXPowGate(global_shift=0.5)
    eq.add_equality_group(iXX ** 0.5, iXX ** 4.5)
    eq.add_equality_group(iXX ** 2.5, iXX ** 6.5)