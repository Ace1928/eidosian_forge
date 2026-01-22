import logging
from sympy.external import import_module
from sympy.testing.pytest import raises, SKIP, warns_deprecated_sympy
import sympy as sy
from sympy.core.singleton import S
from sympy.abc import x, y, z, t
from sympy.printing.theanocode import (theano_code, dim_handling,
def test_MatMul():
    expr = X * Y * Z
    expr_t = theano_code_(expr)
    assert isinstance(expr_t.owner.op, tt.Dot)
    assert theq(expr_t, Xt.dot(Yt).dot(Zt))