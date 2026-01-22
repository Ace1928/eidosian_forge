import logging
from sympy.external import import_module
from sympy.testing.pytest import raises, SKIP, warns_deprecated_sympy
import sympy as sy
from sympy.core.singleton import S
from sympy.abc import x, y, z, t
from sympy.printing.theanocode import (theano_code, dim_handling,
def test_global_cache():
    """ Test use of the global cache. """
    from sympy.printing.theanocode import global_cache
    backup = dict(global_cache)
    try:
        global_cache.clear()
        for s in [x, X, f_t]:
            with warns_deprecated_sympy():
                st = theano_code(s)
                assert theano_code(s) is st
    finally:
        global_cache.update(backup)