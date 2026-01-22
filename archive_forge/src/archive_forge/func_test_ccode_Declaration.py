from sympy.core import (
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)
from sympy.functions import (
from sympy.sets import Range
from sympy.logic import ITE, Implies, Equivalent
from sympy.codegen import For, aug_assign, Assignment
from sympy.testing.pytest import raises, XFAIL
from sympy.printing.c import C89CodePrinter, C99CodePrinter, get_math_macros
from sympy.codegen.ast import (
from sympy.codegen.cfunctions import expm1, log1p, exp2, log2, fma, log10, Cbrt, hypot, Sqrt
from sympy.codegen.cnodes import restrict
from sympy.utilities.lambdify import implemented_function
from sympy.tensor import IndexedBase, Idx
from sympy.matrices import Matrix, MatrixSymbol, SparseMatrix
from sympy.printing.codeprinter import ccode
def test_ccode_Declaration():
    i = symbols('i', integer=True)
    var1 = Variable(i, type=Type.from_expr(i))
    dcl1 = Declaration(var1)
    assert ccode(dcl1) == 'int i'
    var2 = Variable(x, type=float32, attrs={value_const})
    dcl2a = Declaration(var2)
    assert ccode(dcl2a) == 'const float x'
    dcl2b = var2.as_Declaration(value=pi)
    assert ccode(dcl2b) == 'const float x = M_PI'
    var3 = Variable(y, type=Type('bool'))
    dcl3 = Declaration(var3)
    printer = C89CodePrinter()
    assert 'stdbool.h' not in printer.headers
    assert printer.doprint(dcl3) == 'bool y'
    assert 'stdbool.h' in printer.headers
    u = symbols('u', real=True)
    ptr4 = Pointer.deduced(u, attrs={pointer_const, restrict})
    dcl4 = Declaration(ptr4)
    assert ccode(dcl4) == 'double * const restrict u'
    var5 = Variable(x, Type('__float128'), attrs={value_const})
    dcl5a = Declaration(var5)
    assert ccode(dcl5a) == 'const __float128 x'
    var5b = Variable(var5.symbol, var5.type, pi, attrs=var5.attrs)
    dcl5b = Declaration(var5b)
    assert ccode(dcl5b) == 'const __float128 x = M_PI'