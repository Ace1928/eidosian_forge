import numpy as np
import pytest
import sympy
import cirq
def test_yy_init():
    assert cirq.YYPowGate(exponent=1).exponent == 1
    v = cirq.YYPowGate(exponent=0.5)
    assert v.exponent == 0.5