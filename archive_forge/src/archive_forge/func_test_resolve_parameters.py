import pytest, sympy
import cirq
from cirq.study import ParamResolver
@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_resolve_parameters(resolve_fn):

    class NoMethod:
        pass

    class ReturnsNotImplemented:

        def _is_parameterized_(self):
            return NotImplemented

        def _resolve_parameters_(self, resolver, recursive):
            return NotImplemented

    class SimpleParameterSwitch:

        def __init__(self, var):
            self.parameter = var

        def _is_parameterized_(self) -> bool:
            return self.parameter != 0

        def _resolve_parameters_(self, resolver: ParamResolver, recursive: bool):
            self.parameter = resolver.value_of(self.parameter, recursive)
            return self
    assert not cirq.is_parameterized(NoMethod())
    assert not cirq.is_parameterized(ReturnsNotImplemented())
    assert cirq.is_parameterized(SimpleParameterSwitch('a'))
    assert not cirq.is_parameterized(SimpleParameterSwitch(0))
    ni = ReturnsNotImplemented()
    d = {'a': 0}
    r = cirq.ParamResolver(d)
    no = NoMethod()
    assert resolve_fn(no, r) == no
    assert resolve_fn(no, d) == no
    assert resolve_fn(ni, r) == ni
    assert resolve_fn(SimpleParameterSwitch(0), r).parameter == 0
    assert resolve_fn(SimpleParameterSwitch('a'), r).parameter == 0
    assert resolve_fn(SimpleParameterSwitch('a'), d).parameter == 0
    assert resolve_fn(sympy.Symbol('a'), r) == 0
    a, b, c = tuple((sympy.Symbol(l) for l in 'abc'))
    x, y, z = (0, 4, 7)
    resolver = {a: x, b: y, c: z}
    assert resolve_fn((a, b, c), resolver) == (x, y, z)
    assert resolve_fn([a, b, c], resolver) == [x, y, z]
    assert resolve_fn((x, y, z), resolver) == (x, y, z)
    assert resolve_fn([x, y, z], resolver) == [x, y, z]
    assert resolve_fn((), resolver) == ()
    assert resolve_fn([], resolver) == []
    assert resolve_fn(1, resolver) == 1
    assert resolve_fn(1.1, resolver) == 1.1
    assert resolve_fn(1j, resolver) == 1j