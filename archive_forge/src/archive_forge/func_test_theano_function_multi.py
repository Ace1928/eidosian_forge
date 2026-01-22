import logging
from sympy.external import import_module
from sympy.testing.pytest import raises, SKIP, warns_deprecated_sympy
import sympy as sy
from sympy.core.singleton import S
from sympy.abc import x, y, z, t
from sympy.printing.theanocode import (theano_code, dim_handling,
def test_theano_function_multi():
    """ Test theano_function() with multiple outputs. """
    f = theano_function_([x, y], [x + y, x - y])
    o1, o2 = f(2, 3)
    assert o1 == 5
    assert o2 == -1