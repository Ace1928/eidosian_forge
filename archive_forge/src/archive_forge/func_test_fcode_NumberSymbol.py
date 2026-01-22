from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import (Function, Lambda, diff)
from sympy.core.mod import Mod
from sympy.core import (Catalan, EulerGamma, GoldenRatio)
from sympy.core.numbers import (E, Float, I, Integer, Rational, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import (conjugate, sign)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (atan2, cos, sin)
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import Integral
from sympy.sets.fancysets import Range
from sympy.codegen import For, Assignment, aug_assign
from sympy.codegen.ast import Declaration, Variable, float32, float64, \
from sympy.core.expr import UnevaluatedExpr
from sympy.core.relational import Relational
from sympy.logic.boolalg import And, Or, Not, Equivalent, Xor
from sympy.matrices import Matrix, MatrixSymbol
from sympy.printing.fortran import fcode, FCodePrinter
from sympy.tensor import IndexedBase, Idx
from sympy.tensor.array.expressions import ArraySymbol, ArrayElement
from sympy.utilities.lambdify import implemented_function
from sympy.testing.pytest import raises
def test_fcode_NumberSymbol():
    prec = 17
    p = FCodePrinter()
    assert fcode(Catalan) == '      parameter (Catalan = %sd0)\n      Catalan' % Catalan.evalf(prec)
    assert fcode(EulerGamma) == '      parameter (EulerGamma = %sd0)\n      EulerGamma' % EulerGamma.evalf(prec)
    assert fcode(E) == '      parameter (E = %sd0)\n      E' % E.evalf(prec)
    assert fcode(GoldenRatio) == '      parameter (GoldenRatio = %sd0)\n      GoldenRatio' % GoldenRatio.evalf(prec)
    assert fcode(pi) == '      parameter (pi = %sd0)\n      pi' % pi.evalf(prec)
    assert fcode(pi, precision=5) == '      parameter (pi = %sd0)\n      pi' % pi.evalf(5)
    assert fcode(Catalan, human=False) == ({(Catalan, p._print(Catalan.evalf(prec)))}, set(), '      Catalan')
    assert fcode(EulerGamma, human=False) == ({(EulerGamma, p._print(EulerGamma.evalf(prec)))}, set(), '      EulerGamma')
    assert fcode(E, human=False) == ({(E, p._print(E.evalf(prec)))}, set(), '      E')
    assert fcode(GoldenRatio, human=False) == ({(GoldenRatio, p._print(GoldenRatio.evalf(prec)))}, set(), '      GoldenRatio')
    assert fcode(pi, human=False) == ({(pi, p._print(pi.evalf(prec)))}, set(), '      pi')
    assert fcode(pi, precision=5, human=False) == ({(pi, p._print(pi.evalf(5)))}, set(), '      pi')