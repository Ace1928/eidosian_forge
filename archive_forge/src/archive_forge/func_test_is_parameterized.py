import pytest, sympy
import cirq
from cirq.study import ParamResolver
def test_is_parameterized():
    a, b = tuple((sympy.Symbol(l) for l in 'ab'))
    x, y = (0, 4)
    assert not cirq.is_parameterized((x, y))
    assert not cirq.is_parameterized([x, y])
    assert cirq.is_parameterized([a, b])
    assert cirq.is_parameterized([a, x])
    assert cirq.is_parameterized((a, b))
    assert cirq.is_parameterized((a, x))
    assert not cirq.is_parameterized(())
    assert not cirq.is_parameterized([])
    assert not cirq.is_parameterized(1)
    assert not cirq.is_parameterized(1.1)
    assert not cirq.is_parameterized(1j)