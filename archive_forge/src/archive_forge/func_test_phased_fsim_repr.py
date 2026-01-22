import numpy as np
import pytest
import sympy
import cirq
def test_phased_fsim_repr():
    f = cirq.PhasedFSimGate(sympy.Symbol('a'), sympy.Symbol('b'), sympy.Symbol('c'), sympy.Symbol('d'), sympy.Symbol('e'))
    cirq.testing.assert_equivalent_repr(f)