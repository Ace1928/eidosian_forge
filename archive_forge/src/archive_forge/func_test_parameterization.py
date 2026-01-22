import numpy as np
import pytest
import sympy
import cirq
def test_parameterization():
    t = sympy.Symbol('t')
    gpt = cirq.GlobalPhaseGate(coefficient=t)
    assert cirq.is_parameterized(gpt)
    assert cirq.parameter_names(gpt) == {'t'}
    assert not cirq.has_unitary(gpt)
    assert gpt.coefficient == t
    assert (gpt ** 2).coefficient == t ** 2