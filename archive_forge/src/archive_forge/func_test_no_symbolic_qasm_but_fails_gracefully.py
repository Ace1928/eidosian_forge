import itertools
import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('sym', [sympy.Symbol('a'), sympy.Symbol('a') + 1])
def test_no_symbolic_qasm_but_fails_gracefully(sym):
    q = cirq.NamedQubit('q')
    v = cirq.PhasedXPowGate(phase_exponent=sym).on(q)
    assert cirq.qasm(v, args=cirq.QasmArgs(), default=None) is None