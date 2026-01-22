from sympy.codegen import Assignment
from sympy.codegen.ast import none
from sympy.codegen.cfunctions import expm1, log1p
from sympy.codegen.scipy_nodes import cosm1
from sympy.codegen.matrix_nodes import MatrixSolve
from sympy.core import Expr, Mod, symbols, Eq, Le, Gt, zoo, oo, Rational, Pow
from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.functions import acos, KroneckerDelta, Piecewise, sign, sqrt, Min, Max, cot, acsch, asec, coth
from sympy.logic import And, Or
from sympy.matrices import SparseMatrix, MatrixSymbol, Identity
from sympy.printing.pycode import (
from sympy.printing.tensorflow import TensorflowPrinter
from sympy.printing.numpy import NumPyPrinter, SciPyPrinter
from sympy.testing.pytest import raises, skip
from sympy.tensor import IndexedBase, Idx
from sympy.tensor.array.expressions.array_expressions import ArraySymbol, ArrayDiagonal, ArrayContraction, ZeroArray, OneArray
from sympy.external import import_module
from sympy.functions.special.gamma_functions import loggamma
def test_airy_prime():
    from sympy.functions.special.bessel import airyaiprime, airybiprime
    expr1 = airyaiprime(x)
    expr2 = airybiprime(x)
    prntr = SciPyPrinter()
    assert prntr.doprint(expr1) == 'scipy.special.airy(x)[1]'
    assert prntr.doprint(expr2) == 'scipy.special.airy(x)[3]'
    prntr = NumPyPrinter()
    assert 'Not supported' in prntr.doprint(expr1)
    assert 'Not supported' in prntr.doprint(expr2)
    prntr = PythonCodePrinter()
    assert 'Not supported' in prntr.doprint(expr1)
    assert 'Not supported' in prntr.doprint(expr2)