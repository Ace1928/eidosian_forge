import pytest, sympy
import cirq
from cirq.study import ParamResolver
def test_parameter_names():
    a, b, c = tuple((sympy.Symbol(l) for l in 'abc'))
    x, y, z = (0, 4, 7)
    assert cirq.parameter_names((a, b, c)) == {'a', 'b', 'c'}
    assert cirq.parameter_names([a, b, c]) == {'a', 'b', 'c'}
    assert cirq.parameter_names((x, y, z)) == set()
    assert cirq.parameter_names([x, y, z]) == set()
    assert cirq.parameter_names(()) == set()
    assert cirq.parameter_names([]) == set()
    assert cirq.parameter_names(1) == set()
    assert cirq.parameter_names(1.1) == set()
    assert cirq.parameter_names(1j) == set()