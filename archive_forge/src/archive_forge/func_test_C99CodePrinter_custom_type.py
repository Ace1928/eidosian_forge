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
def test_C99CodePrinter_custom_type():
    f128 = FloatType('_Float128', float128.nbits, float128.nmant, float128.nexp)
    p128 = C99CodePrinter({'type_aliases': {real: f128}, 'type_literal_suffixes': {f128: 'Q'}, 'type_func_suffixes': {f128: 'f128'}, 'type_math_macro_suffixes': {real: 'f128', f128: 'f128'}, 'type_macros': {f128: ('__STDC_WANT_IEC_60559_TYPES_EXT__',)}})
    assert p128.doprint(x) == 'x'
    assert not p128.headers
    assert not p128.libraries
    assert not p128.macros
    assert p128.doprint(2.0) == '2.0Q'
    assert not p128.headers
    assert not p128.libraries
    assert p128.macros == {'__STDC_WANT_IEC_60559_TYPES_EXT__'}
    assert p128.doprint(Rational(1, 2)) == '1.0Q/2.0Q'
    assert p128.doprint(sin(x)) == 'sinf128(x)'
    assert p128.doprint(cos(2.0, evaluate=False)) == 'cosf128(2.0Q)'
    assert p128.doprint(x ** (-1.0)) == '1.0Q/x'
    var5 = Variable(x, f128, attrs={value_const})
    dcl5a = Declaration(var5)
    assert ccode(dcl5a) == 'const _Float128 x'
    var5b = Variable(x, f128, pi, attrs={value_const})
    dcl5b = Declaration(var5b)
    assert p128.doprint(dcl5b) == 'const _Float128 x = M_PIf128'
    var5b = Variable(x, f128, value=Catalan.evalf(38), attrs={value_const})
    dcl5c = Declaration(var5b)
    assert p128.doprint(dcl5c) == 'const _Float128 x = %sQ' % Catalan.evalf(f128.decimal_dig)