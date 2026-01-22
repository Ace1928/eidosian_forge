import logging
from sympy.external import import_module
from sympy.testing.pytest import raises, SKIP, warns_deprecated_sympy
import sympy as sy
from sympy.core.singleton import S
from sympy.abc import x, y, z, t
from sympy.printing.theanocode import (theano_code, dim_handling,
def test_example_symbols():
    """
    Check that the example symbols in this module print to their Theano
    equivalents, as many of the other tests depend on this.
    """
    assert theq(xt, theano_code_(x))
    assert theq(yt, theano_code_(y))
    assert theq(zt, theano_code_(z))
    assert theq(Xt, theano_code_(X))
    assert theq(Yt, theano_code_(Y))
    assert theq(Zt, theano_code_(Z))