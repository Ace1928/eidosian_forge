import logging
from sympy.external import import_module
from sympy.testing.pytest import raises, SKIP, warns_deprecated_sympy
import sympy as sy
from sympy.core.singleton import S
from sympy.abc import x, y, z, t
from sympy.printing.theanocode import (theano_code, dim_handling,
def test_complexfunctions():
    with warns_deprecated_sympy():
        xt, yt = (theano_code_(x, dtypes={x: 'complex128'}), theano_code_(y, dtypes={y: 'complex128'}))
    from sympy.functions.elementary.complexes import conjugate
    from theano.tensor import as_tensor_variable as atv
    from theano.tensor import complex as cplx
    with warns_deprecated_sympy():
        assert theq(theano_code_(y * conjugate(x)), yt * xt.conj())
        assert theq(theano_code_((1 + 2j) * x), xt * (atv(1.0) + atv(2.0) * cplx(0, 1)))