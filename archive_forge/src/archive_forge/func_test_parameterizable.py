import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
def test_parameterizable():
    s = sympy.Symbol('s')
    q0 = cirq.LineQubit(0)
    op = cirq.X(q0).with_classical_controls('a')
    opa = cirq.XPowGate(exponent=s).on(q0).with_classical_controls('a')
    assert cirq.is_parameterized(opa)
    assert not cirq.is_parameterized(op)
    assert cirq.resolve_parameters(opa, cirq.ParamResolver({'s': 1})) == op