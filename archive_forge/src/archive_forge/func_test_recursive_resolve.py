import pytest, sympy
import cirq
from cirq.study import ParamResolver
def test_recursive_resolve():
    a, b, c = [sympy.Symbol(l) for l in 'abc']
    resolver = cirq.ParamResolver({a: b + 3, b: c + 2, c: 1})
    assert cirq.resolve_parameters_once(a, resolver) == b + 3
    assert cirq.resolve_parameters(a, resolver) == 6
    assert cirq.resolve_parameters_once(b, resolver) == c + 2
    assert cirq.resolve_parameters(b, resolver) == 3
    assert cirq.resolve_parameters_once(c, resolver) == 1
    assert cirq.resolve_parameters(c, resolver) == 1
    assert cirq.resolve_parameters_once([a, b], {a: b, b: c}) == [b, c]
    assert cirq.resolve_parameters_once(a, {}) == a
    resolver = cirq.ParamResolver({a: b, b: a})
    assert cirq.resolve_parameters_once(a, resolver) == b
    with pytest.raises(RecursionError):
        _ = cirq.resolve_parameters(a, resolver)