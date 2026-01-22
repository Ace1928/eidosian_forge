import os
from tempfile import TemporaryDirectory
from sympy.concrete.summations import Sum
from sympy.core.numbers import (I, oo, pi)
from sympy.core.relational import Ne
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import (LambertW, exp, exp_polar, log)
from sympy.functions.elementary.miscellaneous import (real_root, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.hyper import meijerg
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import And
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.external import import_module
from sympy.plotting.plot import (
from sympy.plotting.plot import (
from sympy.testing.pytest import skip, raises, warns, warns_deprecated_sympy
from sympy.utilities import lambdify as lambdify_
from sympy.utilities.exceptions import ignore_warnings
def test_issue_20113():
    if not matplotlib:
        skip('Matplotlib not the default backend')
    x = Symbol('x')
    with raises(TypeError):
        plot(sin(x), backend=Plot, show=False)
    p2 = plot(sin(x), backend=MatplotlibBackend, show=False)
    assert p2.backend == MatplotlibBackend
    assert len(p2[0].get_data()[0]) >= 30
    p3 = plot(sin(x), backend=DummyBackendOk, show=False)
    assert p3.backend == DummyBackendOk
    assert len(p3[0].get_data()[0]) >= 30
    p4 = plot(sin(x), backend=DummyBackendNotOk, show=False)
    assert p4.backend == DummyBackendNotOk
    assert len(p4[0].get_data()[0]) >= 30
    with raises(NotImplementedError):
        p4.show()
    with raises(NotImplementedError):
        p4.save('test/path')
    with raises(NotImplementedError):
        p4._backend.close()