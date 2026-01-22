import fractions
import numpy as np
import pytest
import sympy
import cirq
def test_recursive_evaluation():
    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    c = sympy.Symbol('c')
    d = sympy.Symbol('d')
    e = sympy.Symbol('e')
    r = cirq.ParamResolver({a: a, b: e + 2, c: b + d, d: a + 3, e: 0})
    assert c.subs(r.param_dict) == b + a + 3
    assert r.value_of(a) == a
    assert sympy.Eq(r.value_of(b), 2)
    assert sympy.Eq(r.value_of(c), a + 5)
    assert sympy.Eq(r.value_of(d), a + 3)
    assert sympy.Eq(r.value_of(e), 0)