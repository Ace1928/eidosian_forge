import logging
from sympy.external import import_module
from sympy.testing.pytest import raises, SKIP, warns_deprecated_sympy
import sympy as sy
from sympy.core.singleton import S
from sympy.abc import x, y, z, t
from sympy.printing.theanocode import (theano_code, dim_handling,
def theq_slice(s1, s2):
    for attr in ['start', 'stop', 'step']:
        a1 = getattr(s1, attr)
        a2 = getattr(s2, attr)
        if a1 is None or a2 is None:
            if not (a1 is None or a2 is None):
                return False
        elif not theq(a1, a2):
            return False
    return True