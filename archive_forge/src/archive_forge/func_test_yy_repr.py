import numpy as np
import pytest
import sympy
import cirq
def test_yy_repr():
    assert repr(cirq.YYPowGate()) == 'cirq.YY'
    assert repr(cirq.YYPowGate(exponent=0.5)) == '(cirq.YY**0.5)'