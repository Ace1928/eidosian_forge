import fractions
import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('p1, p2, p3', [({'a': 1}, {}, {}), ({}, {'a': 1}, {}), ({}, {}, {'a': 1}), ({'a': 'b'}, {}, {'b': 'c'}), ({'a': 'b'}, {'c': 'd'}, {'b': 'c'}), ({'a': 'b'}, {'c': 'a'}, {'b': 'd'}), ({'a': 'b'}, {'c': 'd', 'd': 1}, {'d': 2}), ({'a': 'b'}, {'c': 'd', 'd': 'a'}, {'b': 2})])
@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_compose_associative(p1, p2, p3, resolve_fn):
    r1, r2, r3 = [cirq.ParamResolver({sympy.Symbol(k): sympy.Symbol(v) if isinstance(v, str) else v for k, v in pd.items()}) for pd in [p1, p2, p3]]
    assert sympy.Eq(resolve_fn(r1, resolve_fn(r2, r3)).param_dict, resolve_fn(resolve_fn(r1, r2), r3).param_dict)