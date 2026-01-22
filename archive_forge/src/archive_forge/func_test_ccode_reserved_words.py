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
def test_ccode_reserved_words():
    x, y = symbols('x, if')
    with raises(ValueError):
        ccode(y ** 2, error_on_reserved=True, standard='C99')
    assert ccode(y ** 2) == 'pow(if_, 2)'
    assert ccode(x * y ** 2, dereference=[y]) == 'pow((*if_), 2)*x'
    assert ccode(y ** 2, reserved_word_suffix='_unreserved') == 'pow(if_unreserved, 2)'