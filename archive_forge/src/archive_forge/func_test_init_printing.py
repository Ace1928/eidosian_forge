from symengine import (ccode, unicode, Symbol, sqrt, Pow, Max, sin, Integer, MutableDenseMatrix)
from symengine.test_utilities import raises
from symengine.printing import CCodePrinter, init_printing
def test_init_printing():
    x = Symbol('x')
    assert x._repr_latex_() is None
    init_printing()
    assert x._repr_latex_() == '$x$'