import numpy as np
import pytest
import sympy
import cirq
def test_fsim_repr():
    f = cirq.FSimGate(sympy.Symbol('a'), sympy.Symbol('b'))
    cirq.testing.assert_equivalent_repr(f)