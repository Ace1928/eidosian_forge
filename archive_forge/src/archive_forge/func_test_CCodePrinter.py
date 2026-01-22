from symengine import (ccode, unicode, Symbol, sqrt, Pow, Max, sin, Integer, MutableDenseMatrix)
from symengine.test_utilities import raises
from symengine.printing import CCodePrinter, init_printing
def test_CCodePrinter():
    x = Symbol('x')
    y = Symbol('y')
    myprinter = CCodePrinter()
    assert myprinter.doprint(1 + x, 'bork') == 'bork = 1 + x;'
    assert myprinter.doprint(1 * x) == 'x'
    assert myprinter.doprint(MutableDenseMatrix(1, 2, [x, y]), 'larry') == 'larry[0] = x;\nlarry[1] = y;'
    raises(TypeError, lambda: myprinter.doprint(sin(x), Integer))
    raises(RuntimeError, lambda: myprinter.doprint(MutableDenseMatrix(1, 2, [x, y])))