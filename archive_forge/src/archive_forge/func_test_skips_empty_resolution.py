import pytest, sympy
import cirq
from cirq.study import ParamResolver
@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_skips_empty_resolution(resolve_fn):

    class Tester:

        def _resolve_parameters_(self, resolver, recursive):
            return 5
    t = Tester()
    assert resolve_fn(t, {}) is t
    assert resolve_fn(t, {'x': 2}) == 5