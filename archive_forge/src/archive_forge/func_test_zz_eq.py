import numpy as np
import pytest
import sympy
import cirq
def test_zz_eq():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq.ZZ, cirq.ZZPowGate(), cirq.ZZPowGate(exponent=1, global_shift=0), cirq.ZZPowGate(exponent=3, global_shift=0))
    eq.add_equality_group(cirq.ZZ ** 0.5, cirq.ZZ ** 2.5, cirq.ZZ ** 4.5)
    eq.add_equality_group(cirq.ZZ ** 0.25, cirq.ZZ ** 2.25, cirq.ZZ ** (-1.75))
    iZZ = cirq.ZZPowGate(global_shift=0.5)
    eq.add_equality_group(iZZ ** 0.5, iZZ ** 4.5)
    eq.add_equality_group(iZZ ** 2.5, iZZ ** 6.5)