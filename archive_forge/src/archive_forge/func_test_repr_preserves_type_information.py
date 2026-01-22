from datetime import timedelta
import pytest
import sympy
import numpy as np
import cirq
from cirq.value import Duration
def test_repr_preserves_type_information():
    t = sympy.Symbol('t')
    assert repr(cirq.Duration(micros=1500)) == 'cirq.Duration(micros=1500)'
    assert repr(cirq.Duration(micros=1500.0)) == 'cirq.Duration(micros=1500.0)'
    assert repr(cirq.Duration(millis=1.5)) == 'cirq.Duration(micros=1500.0)'
    assert repr(cirq.Duration(micros=1500 * t)) == "cirq.Duration(micros=sympy.Mul(sympy.Integer(1500), sympy.Symbol('t')))"
    assert repr(cirq.Duration(micros=1500.0 * t)) == "cirq.Duration(micros=sympy.Mul(sympy.Float('1500.0', precision=53), sympy.Symbol('t')))"
    assert repr(cirq.Duration(millis=1.5 * t)) == "cirq.Duration(micros=sympy.Mul(sympy.Float('1500.0', precision=53), sympy.Symbol('t')))"